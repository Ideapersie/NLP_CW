# MNIST & EMNIST CNN Classifier

This repository contains Python scripts to build, train, and evaluate a Convolutional Neural Network (CNN) on handwritten digit datasets. The model is trained on the classic MNIST dataset and then evaluated on both the MNIST test set and the extended EMNIST (digits) dataset to test generalization. 

A key feature of this project is its evaluation script, which identifies misclassified images and generates visual PDF reports showing the true label versus the model's predicted probabilities.

## Features
* **Custom CNN Architecture**: Implements a 3-block Convolutional Neural Network using TensorFlow/Keras with Batch Normalization and Dropout layers to prevent overfitting.
* **Data Augmentation**: Uses `ImageDataGenerator` to apply random rotations, zooms, and shifts during training to make the model robust against distorted handwriting.
* **Dual Testing**: Evaluates the saved model against both MNIST and EMNIST datasets.
* **Misclassification Analysis**: Automatically extracts misclassified examples and plots them alongside a bar chart of the model's prediction probabilities, saving them as PDF files (`misclassified.pdf` and `misclassified_EMNIST.pdf`).

## Prerequisites

To run this project, you will need Python 3.x installed along with the following libraries:
* `tensorflow`
* `numpy`
* `matplotlib`

You can install all dependencies easily using:
```bash
pip install -r requirements.txt
```
## Dataset Requirements
**MNIST**: The MNIST dataset is downloaded automatically via tensorflow.keras.datasets.mnist when you run the scripts.

**EMNIST**: The evaluation script (MNIST_testscript.py) expects the EMNIST digits dataset to be stored locally.

Download the EMNIST dataset (idx format).

Extract it and place the files inside a folder named archive/emnist_source_files in the root directory.

### Usage
1. Train the Model
To train the CNN from scratch, run the training script:

```bash
python MNIST_script.py
```
This will download the MNIST data, train the model for 20 epochs using data augmentation, and save the trained weights as my_NN_3blocks.h5.

2. Evaluate the Model
Once you have a trained model (.h5 file), you can evaluate it and generate the misclassification reports:

```bash
python MNIST_testscript.py
```
This script will load the model, evaluate it on MNIST and EMNIST datasets, print the accuracy, and generate misclassified.pdf and misclassified_EMNIST.pdf in your project folder.