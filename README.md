# CapsNet - TensorFlow

Playground for experimentation with CapsNet.

To initiate the training, use the command:
```
python capsnet.py -t -s -c -v --batchSize 500 --trainingEpochs 100 -d MNIST
```
where -t stands for training, -s for training from scratch, -c stands for model testing after training, -v is used for tensorboard output, and -d specifies the dataset to be used. The system supports two datasets at this point (MNIST and CIFAR-10). Keep in mind that enabling the tensorboard visualization will take up much more GPU memory, hence, might require you to reduce the batch size.

For just testing a trained model, use the command:
```
python capsnet.py -c --batchSize 500 -d MNIST
```
where absence of -t flag indicates that only testing of the trained model has to be performed.

To try CapsNet without routing by agreement (only capsules), execute the capsnet_no_rba.py script as follows:
```
python capsnet_no_rba.py -t -s -c -v --batchSize 500 --trainingEpochs 20 -d MNIST
```

<h2>References:</h2>
<ol>
<li>https://github.com/ageron/handson-ml/blob/master/extra_capsnets.ipynb</li>
</ol>

<br/> Author: <b>Shoaib Ahmed Siddiqui</b>
<br/> Email: <b>shoaib_ahmed.siddiqui@dfki.de</b>
