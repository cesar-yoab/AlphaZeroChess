#include <iostream>
#include "model.h"

int main(int argc, const char* argv[]) {
    torch::manual_seed(123);

    // Create a device we pass around based on whether CUDA is available
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    }

    std::cout << "Starting dimensions test..." << std::endl;

    torch::Tensor t = torch::rand({128, 73, 8, 8}).to(device);
    PolicyValueNet<ResBlock> azero = AZeroNet();
    azero.to(device);

    std::pair<torch::Tensor, torch::Tensor> policy_value = azero.forward(t);
    torch::Tensor p = policy_value.first;
    torch::Tensor v = policy_value.second;


    std::cout << "Policy size: " << p.sizes() << std::endl;
    std::cout << "Value size: " << v.sizes() << std::endl;
    return 0;
}
