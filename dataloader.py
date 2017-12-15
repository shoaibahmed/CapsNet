import os
import sys
import wget
import tarfile

import numpy as np
import tensorflow as tf

if sys.version_info[0] == 3:
	print ("Using Python 3")
	import pickle
else:
	print ("Using Python 2")
	import cPickle as pickle

# Import dataset files
from tensorflow.examples.tutorials.mnist import input_data
datasetDict = {"MNIST": 1, "CIFAR-10": 2}

# Read the set file
def loadDataset(rootDirectory, fileList):
	# Load all the files
	finalDict = {"data": None, "labels": None}
	for fileName in fileList:
		fileName = os.path.join(rootDirectory, fileName)
		print ("Reading data from file: %s" % (fileName))
		with open(fileName, 'rb') as fo:
			if sys.version_info[0] == 3:
				dict = pickle.load(fo, encoding='bytes')
			else:
				dict = pickle.load(fo)

		if finalDict["data"] is None:
			finalDict["data"] = dict["data"]
			finalDict["labels"] = dict["labels"]
		else:
			finalDict["data"] = np.vstack([finalDict["data"], dict["data"]])
			finalDict["labels"] += dict["labels"]

	finalDict["data"] = np.array(finalDict["data"])
	finalDict["labels"] = np.array(finalDict["labels"])
	return finalDict

def downloadAndExtractDataset(url):
	print ("Downloading dataset from URL: %s" % (url))
	filename = wget.download(url)

	# Extract the tar file
	print ("Extracting archive")
	tar = tarfile.open(filename)
	tar.extractall()
	tar.close()

	print ("Dataset successfully downloaded!")

# Currently only supported for CIFAR-10
class Set():
	def __init__(self, images, labels, one_hot=False):
		# self.data = data
		self.images = images
		self.labels = labels
		self.epochs_completed = 0
		self.index_in_epoch = 0

		self.num_examples = self.labels.shape[0]

	def next_batch(self, batch_size, shuffle=True):
		"""Return the next `batch_size` examples from this data set."""
		start = self.index_in_epoch
		# Shuffle for the first epoch
		if self.epochs_completed == 0 and start == 0 and shuffle:
			perm0 = np.arange(self.num_examples)
			np.random.shuffle(perm0)
			self.images = self.images[perm0]
			self.labels = self.labels[perm0]

		# Go to the next epoch
		if start + batch_size > self.num_examples:
			# Finished epoch
			self.epochs_completed += 1
			# Get the rest examples in this epoch
			rest_num_examples = self.num_examples - start
			images_rest_part = self.images[start:self.num_examples]
			labels_rest_part = self.labels[start:self.num_examples]

			# Shuffle the data
			if shuffle:
				perm = np.arange(self.num_examples)
				np.random.shuffle(perm)
				self.images = self.images[perm]
				self.labels = self.labels[perm]

			# Start next epoch
			start = 0
			self.index_in_epoch = batch_size - rest_num_examples
			end = self.index_in_epoch
			images_new_part = self.images[start:end]
			labels_new_part = self.labels[start:end]
			return [np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)]

		else:
			self.index_in_epoch += batch_size
			end = self.index_in_epoch
			return [self.images[start:end], self.labels[start:end]]

	@property
	def num_examples(self):
		return self.num_examples

class DataLoader():
	def __init__(self, datasetName, one_hot=False):
		self.datasetName = datasetName
		if self.datasetName == "MNIST":
			mnist = input_data.read_data_sets('MNIST_data', one_hot=one_hot)
			self.train = mnist.train
			self.test = mnist.test
			self.validation = mnist.validation
		elif self.datasetName == "CIFAR-10":
			cifarDatasetDirectory = 'cifar-10-batches-py'
			url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
			if not os.path.exists(cifarDatasetDirectory):
				downloadAndExtractDataset(url)
			data = loadDataset(cifarDatasetDirectory, ["data_batch_" + str(i) for i in range(1, 6)] + ["test_batch"])
			for key in data:
				print ("Key: %s | Data shape: %s" % (key, str(data[key].shape)))
			numTrainingExamples = 45000
			numValidationExamples = 5000
			numTestExamples = 10000
			self.train = Set(data["data"][:numTrainingExamples], data["labels"][:numTrainingExamples])
			self.validation = Set(data["data"][numTrainingExamples:numTrainingExamples+numValidationExamples], 
				data["labels"][numTrainingExamples:numTrainingExamples+numValidationExamples])
			self.test = Set(data["data"][numTrainingExamples+numValidationExamples:numTrainingExamples+numValidationExamples+numTestExamples], 
				data["labels"][numTrainingExamples+numValidationExamples:numTrainingExamples+numValidationExamples+numTestExamples])

		else:
			print ("Error: Dataset not found within the specified list")
			exit (-1)

		print ("Training examples:", self.train.num_examples)
		print ("Test examples:", self.test.num_examples)
		print ("Validation examples:", self.validation.num_examples)

		print ("Train:", self.train.images.shape, self.train.labels.shape)
		print ("Validation:", self.validation.images.shape, self.validation.labels.shape)
		print ("Test:", self.test.images.shape, self.test.labels.shape)

if __name__ == "__main__":
	print ("Executing main")
	# data = DataLoader(datasetName = "MNIST")
	data = DataLoader(datasetName = "CIFAR-10")

	# Test by obtaining a data sample
	batch = data.train.next_batch(batch_size=2)
	print ("Images:", batch[0].shape)
	print ("Labels:", batch[1].shape)

	from PIL import Image
	j = Image.fromarray(batch[0][0, :].reshape([32, 32, 3]))
	j.save("out.png")