# CapsNet - TensorFlow

Playground for experimentation with CapsNet.

To initiate the training, use the command:
```
python capsnet.py -t -s -c -v --batchSize 500 --trainingEpochs 100 -d MNIST
```
where -t stands for training, -s for training from scratch, -c stands for model testing after training, -v is used for tensorboard output, and -d specifies the dataset to be used. The system supports two datasets at this point (MNIST and CIFAR-10). Keep in mind that enabling the tensorboard visualization will take up much more GPU memory, hence, might require you to reduce the batch size.

<h2>References:</h2>
<ol>
<li>https://github.com/ageron/handson-ml/blob/master/extra_capsnets.ipynb</li>
</ol>

<br/> Author: <b>Shoaib Ahmed Siddiqui</b>
<br/> Email: <b>12bscsssiddiqui@seecs.edu.pk</b>
