import torch
import torchvision

dummy_input = torch.randn(1, 3, 224, 224)
model = torchvision.models.alexnet(pretrained=True)

input_names = ["input1"]
output_names = ["output1"]

torch.onnx.export(
  model,
  dummy_input,
  "alexnet.onnx",
  verbose=True,
  input_names=input_names,
  output_names=output_names
)