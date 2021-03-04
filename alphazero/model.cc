/*
 * Implementation of the policy-value network used in the AlphaZero algorithm.
 * This implementation was inspired by:
 *      https://github.com/Keson96/ResNet_LibTorch
 */
#include "model.h"

torch::nn::Conv2dOptions conv_options(int64_t in_channels, int64_t out_channels, int64_t kernel,
                                      int64_t strd, int64_t padd) {
    torch::nn::Conv2dOptions options = torch::nn::Conv2dOptions(in_channels, out_channels, kernel);
    options.stride(strd);
    options.padding(padd);
    return options;
}

PolicyValueNet<ResBlock> AZeroNet() {
    // The original AlphaZero network has input with
    // 73 channels and 19 residual blocks.
    PolicyValueNet<ResBlock> model(73, 19);

    return model;
}