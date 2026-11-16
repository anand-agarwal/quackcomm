#!/usr/bin/env python3
"""Summarize a drive recorded with ./demo.sh record.

Prints a timeline (one row per --step seconds) of what the lane detector saw,
what the controller decided and what the safety supervisor let through, then
totals. Usage: rosrun purepursuit bag_summary.py <bag> [--step 0.25]
"""

import argparse
import collections
import sys

import rosbag


def latest_before(series, t):
    """Last (time, msg) in a time-sorted series at or before t, else None."""
    lo, hi = 0, len(series)
    while lo < hi:
        mid = (lo + hi) // 2
        if series[mid][0] <= t:
            lo = mid + 1
        else:
            hi = mid
    return series[lo - 1][1] if lo else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bag")
    parser.add_argument("--step", type=float, default=0.25)
    args = parser.parse_args()

    series = collections.defaultdict(list)
    counts = collections.Counter()
    with rosbag.Bag(args.bag) as bag:
        for topic, msg, stamp in bag.read_messages():
            name = topic.split("/", 2)[-1]
            series[name].append((stamp.to_sec(), msg))
            counts[name] += 1
    if not series:
        print("empty bag")
        return 1
    start = min(s[0][0] for s in series.values())
    end = max(s[-1][0] for s in series.values())
    print("%s: %.1f s" % (args.bag, end - start))
    for name in sorted(counts):
        print("  %-42s %5d msgs  %5.1f Hz" % (name, counts[name], counts[name] / max(end - start, 1e-3)))

    est_s = series.get("purepursuit/lane_estimate", [])
    ctl_s = series.get("purepursuit/controller_status", [])
    saf_s = series.get("purepursuit/safety_state", [])
    print("\n   t    | lane estimate                               | controller                         | safety")
    t = start
    while t <= end:
        est = latest_before(est_s, t)
        ctl = latest_before(ctl_s, t)
        saf = latest_before(saf_s, t)
        e = "-" if est is None else "%-11s %s d=%+.2f phi=%+.2f k=%+.1f %3.0fms" % (
            est.source, "ok " if est.valid else "BAD", est.d, est.phi, est.curvature, est.process_ms)
        c = "-" if ctl is None else "%-11s v=%.2f w=%+.2f k=%+.1f" % (
            ctl.status, ctl.desired_v, ctl.desired_omega, ctl.curvature)
        s = "-" if saf is None else "%-8s v=%.2f w=%+.2f" % (saf.state, saf.authorized_v, saf.authorized_omega)
        print("%7.2f | %-43s | %-34s | %s" % (t - start, e, c, s))
        t += args.step

    def tally(name, key):
        return collections.Counter(key(m) for _, m in series.get(name, []))

    print("\nlane estimate sources:", dict(tally("purepursuit/lane_estimate",
                                                 lambda m: m.source if m.valid else "BAD:" + m.source)))
    print("controller states:     ", dict(tally("purepursuit/controller_status", lambda m: m.status)))
    print("safety states:         ", dict(tally("purepursuit/safety_state", lambda m: "%s (%s)" % (m.state, m.reason))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
