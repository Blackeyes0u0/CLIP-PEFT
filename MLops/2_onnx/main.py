import torch
import torchvision.models as models


model = models.resnet18(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch_output = model(dummy_input)

torch.onnx.export(model, dummy_input, "resnet18.onnx")


import onnx

model = onnx.load("resnet18.onnx")

print(onnx.helper.printable_graph(model.graph))


import numpy as np
import onnxruntime as ort

ort_session = ort.InferenceSession("resnet18.onnx")

outputs = ort_session.run(None, {"input.1": np.random.randn(1, 3, 224, 224).astype(np.float32)})

ort_outputs = ort_session.run(None, {"input.1": dummy_input.numpy()})

print(outputs)


# validation model with margin of error
np.testing.assert_allclose(torch_output.detach().numpy(), ort_outputs[0], rtol=1e-03, atol=1e-05)
