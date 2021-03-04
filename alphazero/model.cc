/*
 * Implementation of the policy-value network used in the AlphaZero algorithm.
 * This implementation was inspired by:
 *      https://github.com/Keson96/ResNet_LibTorch
 */
#include <torch/torch.h>

torch::nn::Conv2dOptions conv_options(int64_t in_channels, int64_t out_channels, int64_t kernel,
                               int64_t strd, int64_t padd) {
    torch::nn::Conv2dOptions conv_options(in_channels, out_channels, kernel);
    conv_options.stride(strd);
    conv_options.padding(padd);
    return conv_options;
}

/*
    Residual block as explained in the AlphaZero paper
    this block consists of
        (1) A convolution of 256 filters, kernel 3x3, stride of 1 and padding of 1
        (2) Batch normalization
        (3) Rectified nonlinearity
        (4) Convolution of 256 filters, kernel 3x3 and stride of 1 and padding of 1
        (5) Batch normalization
        (6) Skip Connection (As in ResNet paper)
        (7) Rectified nonlinearity
 */

struct ResBlock : torch::nn::Module {
    torch::nn::Sequential res_block;

    ResBlock(int64_t in_channels) : res_block(
            torch::nn::Conv2d(conv_options(in_channels, 256, 3, 1, 1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::Functional(torch::relu),
            torch::nn::Conv2d(conv_options(256, 256, 3, 1, 1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::Functional(torch::relu)
            ) {
        register_module("block", res_block);
    }

    torch::Tensor forward(torch::Tensor input) {
        at::Tensor residual(input.clone());

        auto x = res_block->forward(input);
        x = torch::relu(x + residual);
        return x;
    }
};

/*
    The policy-value network. Given a state s it follows a sequentially through
    an initial block:
        (1) Convolution 256 filters, kernel 3x3, stride of 1 and padding of 1
        (2) Batch normalization
        (3) Rectifier nonlinearity
    This is followed by a residual tower consisting of 19 residuals blocks as
    described in the class ResBlock. This output is then used for two different heads
    the first head computes the policy and passes through:
        (1) Convolution 256 filters, kernel 3x3, stride of 1 and padding of 1
        (2) Batch normalization
        (3) Rectifier nonlinearity
        (4) Convolution 73 filters, kernel 3x3 stride of 1 and padding of 1
    The second head goes through a block which computes:
        (1) Convolution 256, kernel 1x1, stride of 1 and padding of 1
        (2) Batch normalization
        (3) Rectifier nonlinearity
        (4) Rectified linear layer of size 256
        (5) tanh-linear layer of size 1
 */

template <class Block> struct PolicyValueNet : torch::nn::Module {

    torch::nn::Sequential init_block;
    torch::nn::Sequential res_tower;
    torch::nn::Sequential policy;
    torch::nn::Sequential value;

    PolicyValueNet(int64_t in_channels, int64_t num_blocks) :
        init_block(_make_init_layer(in_channels)),
        res_tower(_make_res_tower(num_blocks, 256)),
        policy(_make_policy_layer()),
        value(_make_value_layer()) {

        register_module("init_block", init_block);
        register_module("res_tower", res_tower);
        register_module("policy", policy);
        register_module("value", value);
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor input) {
        auto x = init_block->forward(input);
        x = res_tower->forward(x);

        // Compute policy and values
        at::Tensor xcpy = x.clone();
        auto p = policy->forward(x);
        auto v = value->forward(xcpy);

        return std::make_pair(p, v);
    }

private:
    torch::nn::Sequential _make_init_layer(int64_t in_channels) {
        torch::nn::Sequential il(
                torch::nn::Conv2d(conv_options(in_channels, 256, 3, 1, 1)),
                torch::nn::BatchNorm2d(256),
                torch::nn::Functional(torch::relu)
        );
        return il;
    }

    torch::nn::Sequential _make_res_tower(const int64_t blocks, int64_t in_channels) {
       torch::nn::Sequential rt;

       for(int64_t i = 0; i <= blocks; ++i) {
          rt->push_back(Block(in_channels));
       }

       return rt;
    }

    torch::nn::Sequential _make_policy_layer() {
        torch::nn::Sequential p(
                torch::nn::Conv2d(conv_options(256, 256, 3, 1, 1)),
                torch::nn::BatchNorm2d(256),
                torch::nn::Functional(torch::relu),
                torch::nn::Conv2d(conv_options(256, 73, 3, 1, 1))
        );

        return p;
    }

    torch::nn::Sequential _make_value_layer() {
        torch::nn::Sequential v(
                torch::nn::Conv2d(conv_options(256, 1, 1, 1, 0)),
                torch::nn::BatchNorm2d(1),
                torch::nn::Functional(torch::relu),
                torch::nn::Flatten(),
                torch::nn::Linear(64, 256),
                torch::nn::Functional(torch::relu),
                torch::nn::Linear(256, 1),
                torch::nn::Functional(torch::tanh)
        );

        return v;
    }

};

PolicyValueNet<ResBlock> AlphaZeroNet() {
    // The original AlphaZero network has input with
    // 73 channels and 19 residual blocks.
    PolicyValueNet<ResBlock> model(73, 19);

    return model;
}