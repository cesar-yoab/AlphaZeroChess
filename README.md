# AlphaZero Chess
An implementation of the AlphaZero algorithm for the game of chess. This implementation uses
Facebooks [libtorch](https://pytorch.org/cppdocs/installing.html) for the neural network and
Deepmind [open_spiel](https://github.com/deepmind/open_spiel) for the chess environment. 
We implement in C++ to take full advantage of threads which is limited in Python, moreover
C++ code is significantly faster for training. The implementation has actors that generate
data through self-play using monte carlo tree search (MCTS) with an evaluator that uses
a neural network, a learner that updates the network based on those games, and evaluators
playing vs standard MCTS to gauge process. Both write checkpoints and logs that can be 
analyzed programmatically.

## Pre-requisites
To run the code in this repository you need to have: 

* C++ 14 compiler (eg. gcc)
* Libtorch
* open_spiel

If you are training on a GPU then you also need:

* CUDA Toolkit
* cuDNN (You need a developer account to get the drivers)

This code was developed on Ubuntu and should work with a few minor changes on 
other linux distributions and OS X, significant changes are required 
to use this code on a Windows machine however.
