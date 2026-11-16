#!/usr/bin/env python3
"""Merge YOLO-format datasets into one Duckietown training set.

Each source is a folder exported in "YOLOv8" format (Roboflow's export, or the
output of extract_frames.py after labelling). Expected layout:

    <source>/data.yaml                 names: [...]  (list or {id: name})
    <source>/{train,valid,val,test}/images/*.jpg
    <source>/{train,valid,val,test}/labels/*.txt

A folder with images/ and labels/ directly under it (no split) also works.
Class names are mapped onto TARGET_CLASSES through ALIASES. Boxes of any other
class are dropped. Images keep their source split (valid/val/test -> val);
unsplit sources are split 85/15 with a fixed seed.

    python3 training/merge_datasets.py ~/quackcomm-training/out/duckietown \
        ~/quackcomm-training/datasets/roboflow/* \
        --alias "rubber duck=duckie"
"""

import argparse
import collections
import os
import random
import shutil
import sys

import yaml

TARGET_CLASSES = ["duckie", "duckiebot", "cone"]

ALIASES = {
    "duckie": "duckie", "duckies": "duckie", "duck": "duckie", "ducks": "duckie",
    "rubber duck": "duckie", "rubber_duck": "duckie", "pedestrian": "duckie",
    "duckiebot": "duckiebot", "duckiebots": "duckiebot", "duckie bot": "duckiebot",
    "duckie_bot": "duckiebot", "bot": "duckiebot", "robot": "duckiebot",
    "cone": "cone", "cones": "cone", "traffic cone": "cone", "traffic_cone": "cone",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
SPLIT_NAMES = {"train": "train", "valid": "val", "val": "val", "test": "val"}


def load_names(source):
    path = os.path.join(source, "data.yaml")
    if not os.path.isfile(path):
        raise SystemExit("%s has no data.yaml" % source)
    with open(path) as handle:
        names = yaml.safe_load(handle).get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=int)]
    if not names:
        raise SystemExit("%s/data.yaml has no names" % source)
    return [str(name) for name in names]


def class_map(names, aliases):
    mapping = {}
    for index, name in enumerate(names):
        target = aliases.get(name.strip().lower())
        if target in TARGET_CLASSES:
            mapping[index] = TARGET_CLASSES.index(target)
    return mapping


def split_dirs(source):
    found = []
    for name, split in SPLIT_NAMES.items():
        images = os.path.join(source, name, "images")
        if os.path.isdir(images):
            found.append((images, os.path.join(source, name, "labels"), split))
    if not found and os.path.isdir(os.path.join(source, "images")):
        found.append((os.path.join(source, "images"), os.path.join(source, "labels"), None))
    return found


def convert_label(path, mapping):
    lines = []
    if not os.path.isfile(path):
        return lines
    with open(path) as handle:
        for raw in handle:
            parts = raw.split()
            if len(parts) < 5:
                continue
            source_id = int(float(parts[0]))
            if source_id not in mapping:
                continue
            if len(parts) == 5:
                box = parts[1:]
            elif len(parts) % 2 == 1:
                # Roboflow exports polygon / 4-corner rows (cls x1 y1 x2 y2 ...)
                # for many boxes; reduce them to their enclosing box.
                coords = [min(max(float(value), 0.0), 1.0) for value in parts[1:]]
                xs, ys = coords[0::2], coords[1::2]
                width, height = max(xs) - min(xs), max(ys) - min(ys)
                if width <= 0 or height <= 0:
                    continue
                box = ["%.6f" % value for value in
                       ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, width, height)]
            else:
                continue
            lines.append(" ".join([str(mapping[source_id])] + box))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output")
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--alias", action="append", default=[], help="extra 'source name=target' mapping")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--keep-empty", type=float, default=0.15,
                        help="fraction of images with no target box to keep as negatives")
    args = parser.parse_args()

    aliases = dict(ALIASES)
    for item in args.alias:
        name, _, target = item.partition("=")
        aliases[name.strip().lower()] = target.strip().lower()

    rng = random.Random(0)
    counts = collections.Counter()
    images_per_split = collections.Counter()
    for split in ("train", "val"):
        os.makedirs(os.path.join(args.output, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.output, "labels", split), exist_ok=True)

    for source in args.sources:
        source = os.path.expanduser(source)
        names = load_names(source)
        mapping = class_map(names, aliases)
        tag = os.path.basename(os.path.normpath(source)).replace(" ", "_")
        dropped = [name for index, name in enumerate(names) if index not in mapping]
        print("%s: %s -> %s; dropped classes: %s" % (
            tag, names, {names[k]: TARGET_CLASSES[v] for k, v in mapping.items()}, dropped or "none"))
        if not mapping:
            print("  no class maps onto %s; use --alias" % TARGET_CLASSES)
            continue
        for images_dir, labels_dir, split in split_dirs(source):
            for filename in sorted(os.listdir(images_dir)):
                stem, extension = os.path.splitext(filename)
                if extension.lower() not in IMAGE_EXTENSIONS:
                    continue
                lines = convert_label(os.path.join(labels_dir, stem + ".txt"), mapping)
                if not lines and rng.random() > args.keep_empty:
                    continue
                target_split = split or ("val" if rng.random() < args.val_fraction else "train")
                out_stem = "%s__%s" % (tag, stem)
                shutil.copyfile(
                    os.path.join(images_dir, filename),
                    os.path.join(args.output, "images", target_split, out_stem + extension.lower()),
                )
                with open(os.path.join(args.output, "labels", target_split, out_stem + ".txt"), "w") as handle:
                    handle.write("\n".join(lines) + ("\n" if lines else ""))
                images_per_split[target_split] += 1
                for line in lines:
                    counts[(target_split, TARGET_CLASSES[int(line.split()[0])])] += 1

    with open(os.path.join(args.output, "data.yaml"), "w") as handle:
        yaml.safe_dump({
            "path": os.path.abspath(args.output),
            "train": "images/train",
            "val": "images/val",
            "names": {index: name for index, name in enumerate(TARGET_CLASSES)},
        }, handle, sort_keys=False)

    print("\nimages: train=%d val=%d" % (images_per_split["train"], images_per_split["val"]))
    for split in ("train", "val"):
        print("  %s boxes: %s" % (split, {name: counts[(split, name)] for name in TARGET_CLASSES}))
    if images_per_split["val"] == 0:
        print("WARNING: no validation images", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
