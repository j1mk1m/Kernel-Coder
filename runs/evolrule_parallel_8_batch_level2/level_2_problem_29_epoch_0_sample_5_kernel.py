# Pseudocode for benchmarking (not part of the answer)
model = Model(in_features, out_features).cuda()
model_new = ModelNew(in_features, out_features).cuda()

inputs = get_inputs()[0].cuda()
# ... rest of the benchmarking code ...