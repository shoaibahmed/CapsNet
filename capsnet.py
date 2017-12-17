import tensorflow as tf
from optparse import OptionParser
import numpy as np
import cv2

import os
import shutil
import dataloader

# For visualization
import matplotlib
import matplotlib.pyplot as plt

# Command line options
parser = OptionParser()

# General settings
parser.add_option("-t", "--trainModel", action="store_true", dest="trainModel", default=False, help="Train model")
parser.add_option("-c", "--testModel", action="store_true", dest="testModel", default=False, help="Test model")
parser.add_option("-s", "--startTrainingFromScratch", action="store_true", dest="startTrainingFromScratch", default=False, help="Start training from scratch")
parser.add_option("-v", "--tensorboardVisualization", action="store_true", dest="tensorboardVisualization", default=False, help="Enable tensorboard visualization")
parser.add_option("-d", "--dataset", action="store", type="string", dest="dataset", default="MNIST", help="Dataset to be used for training")

parser.add_option("--imageWidth", action="store", type="int", dest="imageWidth", default=28, help="Image width for feeding into the network")
parser.add_option("--imageHeight", action="store", type="int", dest="imageHeight", default=28, help="Image height for feeding into the network")
parser.add_option("--imageChannels", action="store", type="int", dest="imageChannels", default=1, help="Number of channels in the image")

parser.add_option("--batchSize", action="store", type="int", dest="batchSize", default=50, help="Batch size")
parser.add_option("--trainingEpochs", action="store", type="int", dest="trainingEpochs", default=10, help="Training epochs")
parser.add_option("--learningRate", action="store", type="float", dest="learningRate", default=1e-3, help="Learning Rate")
parser.add_option("--numClasses", action="store", type="int", dest="numClasses", default=10, help="Number of classes")

parser.add_option("--validationStep", action="store", type="int", dest="validationStep", default=10, help="Number of iterations before performing validation")
parser.add_option("--saveStep", action="store", type="int", dest="saveStep", default=1000, help="Number of iterations before saving the model")

# Network params
parser.add_option("--numPrimaryCapsules", action="store", type="int", dest="numPrimaryCapsules", default=32, help="Number of primary capsules")
parser.add_option("--numLevelTwoCapsules", action="store", type="int", dest="numLevelTwoCapsules", default=10, help="Number of level two capsules")
parser.add_option("--primaryCapsulesFeatureSize", action="store", type="int", dest="primaryCapsulesFeatureSize", default=8, help="Dimensionality of the features from the primary capsule")
parser.add_option("--levelTwoCapsulesFeatureSize", action="store", type="int", dest="levelTwoCapsulesFeatureSize", default=16, help="Dimensionality of the features from the level two capsule")
parser.add_option("--epsilon", action="store", type="float", dest="epsilon", default=1e-7, help="Epsilon to avoid numerical issues")

# Regularization
parser.add_option("--numDecoderFirstHiddenLayerUnits", action="store", type="int", dest="numDecoderFirstHiddenLayerUnits", default=512, help="Number of neurons in the first layer of the decoder")
parser.add_option("--numDecoderSecondHiddenLayerUnits", action="store", type="int", dest="numDecoderSecondHiddenLayerUnits", default=1024, help="Number of neurons in the second layer of the decoder")

# Loss params
parser.add_option("--mPlus", action="store", type="float", dest="mPlus", default=0.9, help="m+ to be used in the computation of the margin loss")
parser.add_option("--mMinus", action="store", type="float", dest="mMinus", default=0.1, help="m- to be used in the computation of the margin loss")
parser.add_option("--lambda", action="store", type="float", dest="lambda_", default=0.5, help="Lambda to be used in the computation of the margin loss")
parser.add_option("--alpha", action="store", type="float", dest="alpha", default=0.0005, help="Alpha to be used to weigh the regularization")

parser.add_option("--numTrainingInstances", action="store", type="int", dest="numTrainingInstances", default=55000, help="Training instances")
parser.add_option("--numValidationInstances", action="store", type="int", dest="numValidationInstances", default=5000, help="Validation instances")
parser.add_option("--numTestInstances", action="store", type="int", dest="numTestInstances", default=10000, help="Test instances")

# Directories
parser.add_option("--checkpointDir", action="store", type="string", dest="checkpointDir", default="./checkpoints", help="Path for storing checkpoints")
parser.add_option("--logDir", action="store", type="string", dest="logDir", default="./logs", help="Path for storing logs")
parser.add_option("--modelName", action="store", type="string", dest="modelName", default="capsnet", help="Name of the model to be stored")

# Parse command line options
(options, args) = parser.parse_args()

# Assert if the dataset is within the predefined possible options
datasetDict = {"MNIST": 1, "CIFAR-10": 2}
if options.dataset not in datasetDict:
	print ("Error: Dataset not within the predefined possible options (%s)" % str(datasetDict.keys()))
	exit (-1)
else:
	# Add the dataset name at the end of the model
	options.logDir = options.logDir + "-" + options.dataset
	options.checkpointDir = options.checkpointDir + "-" + options.dataset

	# Import dataset
	dataloader = dataloader.DataLoader(options.dataset, one_hot=False)
	options.numTrainingInstances = dataloader.train.num_examples
	options.numValidationInstances = dataloader.validation.num_examples
	options.numTestInstances = dataloader.test.num_examples
	print ("Training Instances: %d | Validation Instances: %d | Test Instances: %d" % 
		(options.numTrainingInstances, options.numValidationInstances, options.numTestInstances))

	options.imageWidth = dataloader.image_width
	options.imageHeight = dataloader.image_height
	options.imageChannels = dataloader.image_channels

	if options.dataset == "CIFAR-10":
		options.numPrimaryCapsules = 64
		options.numDecoderFirstHiddenLayerUnits = 2048
		options.numDecoderSecondHiddenLayerUnits = 4096

# Print the finalized options
print (options)

# Make sure the batch size is perfectly divisible by the number of instances for training, validation and test
if (options.numTrainingInstances % options.batchSize != 0):
	print ("Error: Batch size not perfectly divisible by the number of training examples (%d)!" % options.numTrainingInstances)
	exit (-1)
if (options.numValidationInstances % options.batchSize != 0):
	print ("Error: Batch size not perfectly divisible by the number of validation examples (%d)!" % options.numValidationInstances)
	exit (-1)
if (options.numTestInstances % options.batchSize != 0):
	print ("Error: Batch size not perfectly divisible by the number of test examples (%d)!" % options.numTestInstances)
	exit (-1)

from enum import Enum
class Dataset(Enum):
	TRAIN = 1
	VALIDATION = 2
	TEST = 3

# Create placeholder
inputPlaceholder = tf.placeholder(tf.float32, shape=[None, int(options.imageHeight * options.imageWidth * options.imageChannels)], name="inputPlaceholder")
reshapedInput = tf.reshape(inputPlaceholder, [-1, options.imageHeight, options.imageWidth, options.imageChannels]) # Output shape: 28 x 28
# labelsPlaceholder = tf.placeholder(tf.int64, shape=[None, options.numClasses], name="labelsPlaceholder")
labelsPlaceholder = tf.placeholder(tf.int64, shape=[None], name="labelsPlaceholder")
containLabelsPlaceholder = tf.placeholder_with_default(False, shape=(), name="containLabelsPlaceholder")

def squashFunction(inputVector, axis=-1, name=None):
	with tf.name_scope(name, default_name='Squash'):
		normSq = tf.reduce_sum(tf.square(inputVector), axis=axis, keep_dims=True)
		norm = tf.sqrt(normSq + options.epsilon)
		squashedOut = (normSq * inputVector) / ((1.0 + normSq) * norm)
		return squashedOut

def initCapsNet(inputPlaceholder):
	net = tf.layers.conv2d(inputs=inputPlaceholder, filters=256, kernel_size=(9, 9), strides=(1, 1), padding='VALID', activation=tf.nn.relu, name='conv1') # Output shape: 20 x 20 x 256
	net = tf.layers.conv2d(inputs=net, filters=options.numPrimaryCapsules * options.primaryCapsulesFeatureSize, kernel_size=(9, 9), 
							strides=(2, 2), padding='VALID', activation=tf.nn.relu, name='conv2') # Output shape: 6 x 6 x 256 (32 8-D Capsules)

	# Reshape to convert 6 x 6 x 256 to 6 x 6 x 32 (8-D Capsules)
	lastLayerShape = list(net.get_shape()) # 6 x 6
	print ("First capsule output size: %s" % str(lastLayerShape))
	totalPrimaryCapsules = int(lastLayerShape[1]) * int(lastLayerShape[2]) * options.numPrimaryCapsules
	net = tf.reshape(net, [-1, totalPrimaryCapsules, options.primaryCapsulesFeatureSize])
	print ("First capsule output size (after reshape): %s" % str(list(net.get_shape())))

	# Apply squashing function to the output of the primary capsules
	caps1Output = squashFunction(net)
	print ("First capsule output size (after squash): %s" % str(list(caps1Output.get_shape())))

	# Wight matrix for routing
	initSigma = 0.01
	initialW = tf.random_normal(shape=(1, totalPrimaryCapsules, options.numLevelTwoCapsules, options.levelTwoCapsulesFeatureSize, options.primaryCapsulesFeatureSize), stddev=initSigma, dtype=tf.float32, name="initialW")
	W = tf.Variable(initialW, name="W")

	batchSize = tf.shape(inputPlaceholder)[0]
	tiledW = tf.tile(W, [batchSize, 1, 1, 1, 1], name="tiledW")

	# Perform the multiplication
	caps1OutputExpanded = tf.expand_dims(caps1Output, -1, name="caps1OutputExpanded")
	caps1OutputTile = tf.expand_dims(caps1OutputExpanded, 2, name="caps1OutputTile")
	caps1OutputTiled = tf.tile(caps1OutputTile, [1, 1, options.numLevelTwoCapsules, 1, 1], name="caps1OutputTiled")

	caps2Predicted = tf.matmul(tiledW, caps1OutputTiled, name="caps2Predicted")

	# Routing by agreement
	rawWeights = tf.zeros([batchSize, totalPrimaryCapsules, options.numLevelTwoCapsules, 1, 1], dtype=np.float32, name="rawWeights")

	# Round 1
	routingWeights = tf.nn.softmax(rawWeights, dim=2, name="routingWeights")

	weightedPredictions = tf.multiply(routingWeights, caps2Predicted, name="weightedPredictions")
	weightedSum = tf.reduce_sum(weightedPredictions, axis=1, keep_dims=True, name="weightedSum")

	caps2OutputRound1 = squashFunction(weightedSum, axis=-2, name="caps2OutputRound1")

	# Round 2
	caps2OutputRound1Tiled = tf.tile(caps2OutputRound1, [1, totalPrimaryCapsules, 1, 1, 1], name="caps2_output_round_1_tiled")

	agreement = tf.matmul(caps2Predicted, caps2OutputRound1Tiled, transpose_a=True, name="agreement")

	rawWeightsRound2 = tf.add(rawWeights, agreement, name="rawWeightsRound2")

	routingWeightsRound2 = tf.nn.softmax(rawWeightsRound2, dim=2, name="routingWeightsRound2")
	weightedPredictionsRound2 = tf.multiply(routingWeightsRound2, caps2Predicted, name="weightedPredictionsRound2")
	weightedSumRound2 = tf.reduce_sum(weightedPredictionsRound2, axis=1, keep_dims=True, name="weightedSumRound2")
	caps2OutputRound2 = squashFunction(weightedSumRound2, axis=-2, name="caps2OutputRound2")

	return caps2OutputRound2

def addDecoder(decoderInput):
	hidden1 = tf.layers.dense(decoderInput, options.numDecoderFirstHiddenLayerUnits, activation=tf.nn.relu, name="hidden1")
	hidden2 = tf.layers.dense(hidden1, options.numDecoderSecondHiddenLayerUnits, activation=tf.nn.relu, name="hidden2")
	decoderOutput = tf.layers.dense(hidden2, options.imageHeight * options.imageWidth * options.imageChannels, activation=tf.nn.sigmoid, name="decoderOutput")
	return decoderOutput

def computeL2Norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
	with tf.name_scope(name, default_name="safe_norm"):
		squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=keep_dims)
		return tf.sqrt(squared_norm + epsilon)

with tf.name_scope('Model'):
	# Create the graph
	caps2Output = initCapsNet(reshapedInput)
	yProbability = computeL2Norm(caps2Output, axis=-2, name="yProbability")
	yProbabilityArgmax = tf.argmax(yProbability, axis=2, name="yProbabilityArgmax")
	predictedY = tf.squeeze(yProbabilityArgmax, axis=[1,2], name="predictedY")

# Add decoder for reconstruction regularization
with tf.name_scope('Decoder'):
	reconstructionTargets = tf.cond(containLabelsPlaceholder, # condition
								 lambda: labelsPlaceholder,        # if True
								 lambda: predictedY,   # if False
								 name="reconstructionTargets")

	reconstructionMask = tf.one_hot(reconstructionTargets, depth=options.numLevelTwoCapsules, name="reconstructionMask")
	reconstructionMaskReshaped = tf.reshape(reconstructionMask, [-1, 1, options.numLevelTwoCapsules, 1, 1], name="reconstructionMaskReshaped")
	caps2OutputMasked = tf.multiply(caps2Output, reconstructionMaskReshaped, name="caps2OutputMasked")

	decoderInput = tf.reshape(caps2OutputMasked, [-1, options.numLevelTwoCapsules * options.levelTwoCapsulesFeatureSize], name="decoderInput")
	decoderOutput = addDecoder(decoderInput)
	reshapedDecoderOutput = tf.reshape(decoderOutput, [-1, options.imageHeight, options.imageWidth, options.imageChannels]) # Output shape: 28 x 28

	squaredDifference = tf.square(inputPlaceholder - decoderOutput, name="squaredDifference")
	reconstructionLoss = tf.reduce_sum(squaredDifference, name="reconstructionLoss")

with tf.name_scope('Loss'):
	caps2OutputNorm = computeL2Norm(caps2Output, axis=-2, keep_dims=True, name="caps2OutputNorm")

	presentErrorRaw = tf.square(tf.maximum(0., options.mPlus - caps2OutputNorm), name="presentErrorRaw")
	presentError = tf.reshape(presentErrorRaw, shape=(-1, 10), name="presentError")

	absentErrorRaw = tf.square(tf.maximum(0., caps2OutputNorm - options.mMinus), name="absentErrorRaw")
	absentError = tf.reshape(absentErrorRaw, shape=(-1, 10), name="absentError")

	T = tf.one_hot(labelsPlaceholder, depth=options.numLevelTwoCapsules, name="T")
	L = tf.add(T * presentError, options.lambda_ * (1.0 - T) * absentError, name="L")
	marginLoss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="marginLoss")

	loss = tf.add(marginLoss, options.alpha * reconstructionLoss, name="loss")

with tf.name_scope('Accuracy'):
	correct = tf.equal(labelsPlaceholder, predictedY, name="correct")
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

with tf.name_scope('Optimizer'):
	# Define Optimizer (use the default params defined in TF as used in paper)
	# optimizationStep = tf.train.AdamOptimizer().minimize(loss)
	optimizer = tf.train.AdamOptimizer()

	# Op to calculate every variable gradient
	gradients = tf.gradients(loss, tf.trainable_variables())
	gradients = list(zip(gradients, tf.trainable_variables()))

	# Op to update all variables according to their gradient
	optimizationStep = optimizer.apply_gradients(grads_and_vars=gradients)

if options.tensorboardVisualization:
	# Create a summary to monitor cost tensor
	tf.summary.scalar("loss", loss)
	tf.summary.scalar("margin loss", marginLoss)
	tf.summary.scalar("reconstruction loss", reconstructionLoss)
	tf.summary.scalar("accuracy", accuracy)

	# Add the image summary of reconstructions
	tf.summary.image("Original Image", reshapedInput)
	tf.summary.image("Reconstructed Image", reshapedDecoderOutput)

	# Create summaries to visualize weights
	for var in tf.trainable_variables():
		tf.summary.histogram(var.name, var)

	# Summarize all gradients
	for grad, var in gradients:
		if grad is not None:
			tf.summary.histogram(var.name + '/gradient', grad)

	# Merge all summaries into a single op
	mergedSummaryOp = tf.summary.merge_all()

# Function for evaluation on any specific dataset
def evaluateDataset(sess, dataset=Dataset.VALIDATION):
	if dataset == Dataset.TRAIN:
		datasetName = "Train"
		totalInstances = options.numTrainingInstances
		numIterations = int(options.numTrainingInstances / options.batchSize)
	elif dataset == Dataset.VALIDATION:
		datasetName = "Validation"
		totalInstances = options.numValidationInstances
		numIterations = int(options.numValidationInstances / options.batchSize)
	elif dataset == Dataset.TEST:
		datasetName = "Test"
		totalInstances = options.numTestInstances
		numIterations = int(options.numTestInstances / options.batchSize)
	else:
		print ("Error: Dataset not found in available options!")
		exit (-1)

	# Evaluate the model over the given instances
	correctlyClassifiedInstances = 0
	averageLoss = 0.0
	for i in range(numIterations):
		if dataset == Dataset.TRAIN:
			batch = dataloader.train.next_batch(options.batchSize)
		elif dataset == Dataset.VALIDATION:
			batch = dataloader.validation.next_batch(options.batchSize)
		elif dataset == Dataset.TEST:
			batch = dataloader.test.next_batch(options.batchSize)

		currentLoss, currentAccuracy, currentPredictions = sess.run([loss, accuracy, predictedY], feed_dict={inputPlaceholder: batch[0], labelsPlaceholder: batch[1]})
		correctlyClassifiedInstances += np.sum(currentPredictions == batch[1])
		averageLoss += currentLoss

		print ("Dataset: %s | Loss: %f | Accuracy: %f" % (datasetName, currentLoss, currentAccuracy))

	# Compute the dataset statistics
	averageLoss = averageLoss / numIterations
	currentAccuracy = float(correctlyClassifiedInstances) / totalInstances
	print ("Dataset: %s | Total images: %d | Correctly classified: %d | Accuracy: %f | Average Loss: %f" % (datasetName, totalInstances, correctlyClassifiedInstances, currentAccuracy, averageLoss))
	return currentAccuracy, averageLoss

# Create the saver object
saver = tf.train.Saver()
initOp = tf.global_variables_initializer()

# GPU config
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
logFile = None

if options.trainModel:
	try:
		with tf.Session(config=config) as sess:
			if options.startTrainingFromScratch:
				sess.run(initOp) # Reinitialize the weights randomly

				# Remove the previous checkpoints and logs
				if os.path.exists(options.checkpointDir):
					shutil.rmtree(options.checkpointDir)
				if os.path.exists(options.logDir):
					shutil.rmtree(options.logDir)
				os.mkdir(options.checkpointDir)
				os.mkdir(options.logDir)

				logFile = open(os.path.join(options.logDir, "performance-log.txt"), 'w')
			else:
				if not tf.train.checkpoint_exists(options.checkpointDir):
					print ("Error: No checkpoints found at: %s" % options.checkpointDir)
					exit(-1)
				saver.restore(sess, options.checkpointDir)
				logFile = open(os.path.join(options.logDir, "performance-log.txt"), 'a')

			if options.tensorboardVisualization:
				# Write the graph to file
				summaryWriter = tf.summary.FileWriter(options.logDir, graph=tf.get_default_graph())

			globalStep = 1
			bestValidationAccuracy = 0.0
			
			totalTrainCorrectInstances = 0
			totalTrainInstances = 0
			averageTrainLoss = 0.0
			for epoch in range(options.trainingEpochs):
				print ("Starting training for epoch # %d" % (epoch + 1))
				numSteps = int(options.numTrainingInstances / options.batchSize)
				for step in range(numSteps):
					batch = dataloader.train.next_batch(options.batchSize)
					if options.tensorboardVisualization:
						currentLoss, currentAccuracy, currentPredictions, _, summary = sess.run([loss, accuracy, predictedY, optimizationStep, mergedSummaryOp], feed_dict={inputPlaceholder: batch[0], labelsPlaceholder: batch[1]})
						summaryWriter.add_summary(summary, globalStep)
					else:
						currentLoss, currentAccuracy, currentPredictions, _ = sess.run([loss, accuracy, predictedY, optimizationStep], feed_dict={inputPlaceholder: batch[0], labelsPlaceholder: batch[1], containLabelsPlaceholder: True})
					print ("Global step: %d | Epoch: %d | Step: %d | Train Loss: %f | Train accuracy: %f" % (globalStep, epoch + 1, step + 1, currentLoss, currentAccuracy))

					# Update the statistics
					totalTrainCorrectInstances += np.sum(currentPredictions == batch[1])
					totalTrainInstances += options.batchSize
					averageTrainLoss += currentLoss

					if globalStep % options.validationStep == 0:
						validationAccuracy, validationLoss = evaluateDataset(sess, Dataset.VALIDATION)
						
						# Write the statistics to file
						averageTrainLoss = averageTrainLoss / options.validationStep
						averageTrainAccuracy = float(totalTrainCorrectInstances) / totalTrainInstances
						logFile.write("%d %f %f %f %f %f %f\n" % (globalStep, averageTrainLoss, averageTrainAccuracy, currentLoss, currentAccuracy, validationLoss, validationAccuracy))

						# Reset the values now
						totalTrainCorrectInstances = 0
						totalTrainInstances = 0
						averageTrainLoss = 0.0
						
						# Save the best model if found
						if validationAccuracy > bestValidationAccuracy:
							print ("New best validation accuracy achieved | Previous best: %f | Current best: %f" % (bestValidationAccuracy, validationAccuracy))
							bestValidationAccuracy = validationAccuracy
							outputModelName = options.modelName + "-best"
							saver.save(sess, os.path.join(options.checkpointDir, outputModelName))
							print ("Best model saved: %s" % (os.path.join(options.checkpointDir, outputModelName)))

					if globalStep % options.saveStep == 0:
						# Save final model weights to disk
						saver.save(sess, os.path.join(options.checkpointDir, options.modelName))
						print ("Model saved: %s" % (os.path.join(options.checkpointDir, options.modelName)))

					globalStep += 1

			logFile.close()

			# Save final model weights to disk
			saver.save(sess, os.path.join(options.checkpointDir, options.modelName))
			print ("Model saved: %s" % (os.path.join(options.checkpointDir, options.modelName)))

	except KeyboardInterrupt:
		print ("Keyboard interrupt occured. Closing log files!")
		if logFile is not None:
			logFile.close()
		exit(-1)

def tweak_pose_parameters(output_vectors, min=-0.5, max=0.5, n_steps=11):
	steps = np.linspace(min, max, n_steps) # -0.25, -0.15, ..., +0.25
	pose_parameters = np.arange(options.levelTwoCapsulesFeatureSize) # 0, 1, ..., 15
	tweaks = np.zeros([options.levelTwoCapsulesFeatureSize, n_steps, 1, 1, 1, options.levelTwoCapsulesFeatureSize, 1])
	tweaks[pose_parameters, :, 0, 0, 0, pose_parameters, 0] = steps
	output_vectors_expanded = output_vectors[np.newaxis, np.newaxis]
	return tweaks + output_vectors_expanded

if options.testModel:
	with tf.Session(config=config) as sess:
		if not tf.train.checkpoint_exists(options.checkpointDir):
			print ("Error: No checkpoints found at: %s" % options.checkpointDir)
			exit(-1)
		modelPath = os.path.join(options.checkpointDir, options.modelName + "-best")
		saver.restore(sess, modelPath)
		print ("Model successfully loaded: %s" % (modelPath))

		# Perform testing on the complete test set
		# trainAccuracy, trainLoss = evaluateDataset(sess, Dataset.TRAIN)
		# validationAccuracy, validationLoss = evaluateDataset(sess, Dataset.VALIDATION)
		# testAccuracy, testLoss = evaluateDataset(sess, Dataset.TEST)

		# print ("Dataset: Train | Loss: %f | Accuracy: %f" % (trainLoss, trainAccuracy))
		# print ("Dataset: Validation | Loss: %f | Accuracy: %f" % (validationLoss, validationAccuracy))
		# print ("Dataset: Test | Loss: %f | Accuracy: %f" % (testLoss, testAccuracy))

		# Check the reconstruction to trace the parameter's influence
		numberOfSamplesForTesting = 5
		# sampleImages = dataloader.test.images[:numberOfSamplesForTesting].reshape([-1, 28, 28, 1])
		sampleImages = dataloader.test.images[:numberOfSamplesForTesting]

		# Get the corresponding image reconstruction output
		caps2OutputValue, decoderOutputValue, predictedYValue = sess.run([caps2Output, decoderOutput, predictedY], 
												feed_dict={inputPlaceholder: sampleImages, labelsPlaceholder: np.array([], dtype=np.int64), containLabelsPlaceholder: False})

		sampleImages = sampleImages.reshape(-1, 28, 28)
		reconstructions = decoderOutputValue.reshape([-1, 28, 28])

		plt.figure(figsize=(numberOfSamplesForTesting * 2, 3))
		for index in range(numberOfSamplesForTesting):
			plt.subplot(1, numberOfSamplesForTesting, index + 1)
			plt.imshow(sampleImages[index], cmap="binary")
			plt.title("Label:" + str(dataloader.test.labels[index]))
			plt.axis("off")

		# plt.show()
		outputLocation = os.path.join(options.logDir, "original-images.png")
		plt.savefig(outputLocation, dpi=300)

		plt.figure(figsize=(numberOfSamplesForTesting * 2, 3))
		for index in range(numberOfSamplesForTesting):
			plt.subplot(1, numberOfSamplesForTesting, index + 1)
			plt.title("Predicted:" + str(predictedYValue[index]))
			plt.imshow(reconstructions[index], cmap="binary")
			plt.axis("off")
			
		# plt.show()
		outputLocation = os.path.join(options.logDir, "reconstructed-images.png")
		plt.savefig(outputLocation, dpi=300)

		# Tweak the outputs now
		n_steps = 11

		tweaked_vectors = tweak_pose_parameters(caps2OutputValue, n_steps=n_steps)
		tweaked_vectors_reshaped = tweaked_vectors.reshape(
			[-1, 1, options.numLevelTwoCapsules, options.levelTwoCapsulesFeatureSize, 1])

		tweak_labels = np.tile(dataloader.test.labels[:numberOfSamplesForTesting], options.levelTwoCapsulesFeatureSize * n_steps)
		decoder_output_value = sess.run(decoderOutput, feed_dict={caps2Output: tweaked_vectors_reshaped, labelsPlaceholder: tweak_labels, containLabelsPlaceholder: True})

		tweak_reconstructions = decoder_output_value.reshape([options.levelTwoCapsulesFeatureSize, n_steps, numberOfSamplesForTesting, 28, 28])

		for dim in range(options.levelTwoCapsulesFeatureSize):
			print("Tweaking output dimension # {}".format(dim))
			plt.figure(figsize=(n_steps / 1.2, numberOfSamplesForTesting / 1.5))
			for row in range(numberOfSamplesForTesting):
				for col in range(n_steps):
					plt.subplot(numberOfSamplesForTesting, n_steps, row * n_steps + col + 1)
					plt.imshow(tweak_reconstructions[dim, col, row], cmap="binary")
					plt.axis("off")
			# plt.show()
			outputLocation = os.path.join(options.logDir, "Dim-" + str(dim) + ".png")
			plt.savefig(outputLocation, dpi=300)

	print ("Model tested successfully!")
