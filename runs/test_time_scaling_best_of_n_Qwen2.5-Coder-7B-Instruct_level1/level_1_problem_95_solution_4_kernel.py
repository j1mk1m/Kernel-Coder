# Test the ModelNew class
model_new = ModelNew().cuda()
predictions, targets = get_inputs()

output = model_new(predictions.cuda(), targets.cuda())
print(output)