#include <iostream>
#include "model.cc"

int main(int argc, const char* argv[]) {
    torch::manual_seed(123);

    // Create the device we pass around based on whether CUDA is available.
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    }

    std::cout << "Hello world" << std::endl;
    return 0;
}
