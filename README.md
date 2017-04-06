This software implements multi-layer Long-Short Term Memory (LSTM) of Recurrent Neural Network (RNN) for word level language model training. The model takes Penn tree bank as dataset to train LSTM network to predict the next word in the sequence. After training, the model could be used to generate word sequence that looks like the original training data.

Main features
Run word embedding on CPU to reduce the usage of GPU
Optimization for LSTM forward and backward to reduce the space usage on GPU (around 100M usage in total)
All parameters adjustable to control the model
Monitoring model perplexity per word for training and validation
Monitoring GPU usage and time used
The final result is around perplexity =120 per word after 20 epochs training

Execution
Install Torch 7 with cunn package and Python 2.7 or higher.
First call data.sh to generate vocabulary (10000 in total) and combine original text file into one big chunk of text file in order to do batch training after.
Then call data4RNN.py to convert .txt file into .npy file that is numpy format for Torch, and convert data from word into index based at the same time.
Final call Training.sh (call Training.lua inside) to train model.
Users need pass file location to run.

Acknowledgements
This work is originally based on  https://github.com/wojciechz/learning_to_execute for LSTM training.
