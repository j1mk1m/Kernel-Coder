import torch
import torch.nn as nn

model_new = ModelNew(in_features, out_features)

inputs = get_inputs()

with torch.no_grad():
    outputs = model_new(inputs[0])

print(outputs.shape)