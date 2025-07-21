import onnx

model = onnx.load('model.onnx')
print("Model outputs:")
for o in model.graph.output:
    name = o.name
    shape = [d.dim_value for d in o.type.tensor_type.shape.dim]
    dtype = o.type.tensor_type.elem_type
    print(f"{name}: shape={shape}, dtype={dtype}")

