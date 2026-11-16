#!/usr/bin/env python3
"""Build a device-local TensorRT engine from the vendored ONNX file.

Run this on the Jetson, not on a laptop. The engine is not portable between
GPUs or TensorRT versions. The demo container started by dts duckiebot demo
has an 800 MB memory limit, which is often too small for the builder. Use a
separate privileged container without that limit, then copy the engine into
the image if you want it available offline.

This script does not download weights.
"""

import os
import sys

from pp_core.paths import default_model_path


def build(onnx_path, engine_path, workspace_bytes=256 * 1024 * 1024):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    explicit = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as handle:
        if not parser.parse(handle.read()):
            for index in range(parser.num_errors):
                print(parser.get_error(index))
            raise RuntimeError("TensorRT could not parse the ONNX file")
    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 builder flag enabled.")
    else:
        print("This platform did not report fast FP16. Building FP32.")
    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT returned an empty engine")
        payload = bytes(serialized)
    else:
        engine = builder.build_cuda_engine(network)
        if engine is None:
            raise RuntimeError("TensorRT returned an empty engine")
        payload = bytes(engine.serialize())
    with open(engine_path, "wb") as handle:
        handle.write(payload)
    print("Wrote %s (%d bytes)" % (engine_path, len(payload)))


def main():
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else default_model_path()
    engine_path = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(onnx_path)[0] + ".engine"
    if not os.path.isfile(onnx_path):
        print("ONNX file not found: %s" % onnx_path)
        return 1
    try:
        build(onnx_path, engine_path)
    except ImportError:
        print("tensorrt is not importable in this Python. JetPack must provide it, and this process must be on the Nano.")
        return 1
    except Exception as exc:
        print("Engine build failed: %s" % exc)
        return 1
    print("The running detector still uses ONNX until a CUDA execution backend is enabled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
