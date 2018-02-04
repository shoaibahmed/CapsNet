import os

import matplotlib.pyplot as plt
import matplotlib as mpl

# Load the file
globalSteps = []
averageTrainLoss = []
averageTrainAccuracy = []
currentStepTrainLoss = []
currentStepTrainAccuracy = []
validationLoss = []
validationAccuracy = []

performanceLogFilePath = os.path.join(os.getcwd(), "logs-no-RBA-MNIST", "performance-log.txt")
with open(performanceLogFilePath, 'r') as performanceLogFile:
	for line in performanceLogFile:
		linePortions = line.split()
		globalSteps.append(int(linePortions[0]))
		averageTrainLoss.append(float(linePortions[1]))
		averageTrainAccuracy.append(float(linePortions[2]))
		currentStepTrainLoss.append(float(linePortions[3]))
		currentStepTrainAccuracy.append(float(linePortions[4]))
		validationLoss.append(float(linePortions[5]))
		validationAccuracy.append(float(linePortions[6]))


fig, ax = plt.subplots()
ax.set_title('CapsNet - MNIST (Loss)')

x = globalSteps
ax.plot(x, averageTrainLoss, 'r', label='Train (Average)', linewidth=2.0)
ax.plot(x, currentStepTrainLoss, 'g', label='Train (Current)', linewidth=2.0)
ax.plot(x, validationLoss, 'b', label='Validation', linewidth=2.0)
ax.legend()

plt.savefig('./loss_capsnet_mnist_no_rba.png', dpi=300)
plt.close('all')

fig, ax = plt.subplots()
ax.set_title('CapsNet - MNIST (Accuracy)')

x = globalSteps
ax.plot(x, averageTrainAccuracy, 'r', label='Train (Average)', linewidth=2.0)
ax.plot(x, currentStepTrainAccuracy, 'g', label='Train (Current)', linewidth=2.0)
ax.plot(x, validationAccuracy, 'b', label='Validation', linewidth=2.0)
ax.legend()

plt.savefig('./accuracy_capsnet_mnist_no_rba.png', dpi=300)
plt.close('all')