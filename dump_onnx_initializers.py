import onnx
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python dump_onnx_initializers.py <model.onnx>")
    sys.exit(1)

model = onnx.load(sys.argv[1])
inits = model.graph.initializer

for init in inits:
    arr = onnx.numpy_helper.to_array(init)
    print(f"{init.name} shape={arr.shape} dtype={arr.dtype}")
    # If it's a scalar or 1d, show values
    if arr.size < 20:
        print(f"  values: {arr}")
    else:
        print(f"  (too large to print)")

