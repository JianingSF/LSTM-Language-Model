#!/bin/sh -ef
scripts="/Users/tang_li/Desktop/RNNLM_Penn/"

trainDataDir=$scripts/trainData4RNN.npy
validDataDir=$scripts/validData4RNN.npy
testDataDir=$scriptst/testData4RNN.npy

LogDir=$ExptDir/logs

inputs_size=10000
output_size=10000
wordEmbedding_size=100
n_epochs=20
batch_size=100
seq_length=20
layers=2
rnn_size=200
init_weight=0.08
learningRate=0.5
learningRateReduce=0.5
min_gain_ratio=0.02

useCuda=1

th $scripts/Training.lua -trainDataDir $trainDataDir -validDataDir $validDataDir -testDataDir $testDataDir -inputs_size $inputs_size -output_size $output_size -wordEmbedding_size $wordEmbedding_size -seq_length $seq_length -batch_size $batch_size -n_epochs $n_epochs -layers $layers -rnn_size $rnn_size -init_weight $init_weight -learningRate $learningRate -learningRateReduce $learningRateReduce -min_gain_ratio $min_gain_ratio -useCuda $useCuda




