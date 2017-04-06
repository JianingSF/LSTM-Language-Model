#!/usr/bin/python
import numpy as np
import sys
import re
import cPickle as pickle
import time
from datetime import date, timedelta
from optparse import OptionParser
import fileinput
import random
import os
import glob
import csv

def vocData(data, voc, index):
	data[voc] = index

def word2Index(dataVoc, fileInData4RNN, fileOutData4RNN):
	fp = open(fileInData4RNN)
	data_list=fp.read().split(" ")
	fp.close()
	data_list = [x for x in data_list if x != '']
	
	for idx in range(len(data_list)):
		if dataVoc.has_key(data_list[idx]) == True:
			data_list[idx] = dataVoc[data_list[idx]]
		else:
			#set unknown words by index 3 <unk>
			print data_list[idx]
			data_list[idx] = '3'
						
	arrayData = np.array([int(x) for x in data_list])
	#save to .npy file
	np.save(fileOutData4RNN, arrayData)

if __name__ == '__main__':
	
	filePath="../RNNLM_Penn/"
	fileVocName=filePath+"vocMap.txt"
	fileOutVocName=filePath+"vocMap.py"
	dataVoc = {}
	fileIn = open(fileVocName)
	for i, line in enumerate(fileIn):
		items=line.strip().split("(-)")
		vocData(dataVoc, items[0], items[1])
	fileIn.close()
	fp = open(fileOutVocName,'wb')
	pickle.dump(dataVoc, fp,protocol=2)
	fp.close()
	
	#change words by index number from voc
	fileInData4RNN=filePath+"train_combine.txt"
	fileOutData4RNN=filePath+"trainData4RNN.npy"
	word2Index(dataVoc, fileInData4RNN, fileOutData4RNN)
	
	fileInData4RNN=filePath+"valid_combine.txt"
	fileOutData4RNN=filePath+"validData4RNN.npy"
	word2Index(dataVoc, fileInData4RNN, fileOutData4RNN)
	
	fileInData4RNN=filePath+"test_combine.txt"
	fileOutData4RNN=filePath+"testData4RNN.npy"
	word2Index(dataVoc, fileInData4RNN, fileOutData4RNN)
	
	
	
	
	
	
	
