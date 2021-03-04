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
