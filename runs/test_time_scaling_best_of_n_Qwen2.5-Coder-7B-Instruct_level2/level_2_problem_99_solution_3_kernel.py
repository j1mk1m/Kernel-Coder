#include <iostream>
#include <torch/torch.h>

int main() {
    // Initialize inputs
    auto batch_size = 1024;
    auto in_features = 8192;
    auto out_features = 8192;

    auto x = torch::randn({batch_size, in_features}).cuda();

    // Create model instance
    auto model = std::make_shared<ModelNew>(in_features, out_features);

    // Forward pass
    auto output = model->forward(x);

    // Print output shape
    std::cout << "Output shape: " << output.sizes() << std::endl;

    return 0;
}