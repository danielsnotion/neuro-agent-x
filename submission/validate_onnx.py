import onnxruntime as ort
import numpy as np

# load ONNX model
session = ort.InferenceSession("submission/ranker.onnx")

# dummy feature vector (same as training)
x = np.random.randn(1, 12).astype(np.float32)

output = session.run(None, {"features": x})

print("ONNX output:", output)