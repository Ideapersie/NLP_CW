import numpy as np
import os
import gzip
from tensorflow.keras.models import load_model

# Load .h5 file of arbitrary name for testing (last if more than one)
print(os.getcwd())

for file in os.listdir(os.getcwd()):
    if file.endswith(".h5"):
        print(file)
        
net = load_model("my_NN_3blocks.h5")
net.summary()
# Determine what type of network this is
input_dims = net.input_shape
netType = 'CNN' if len(input_dims) > 2 else 'MLP'

# Test with MNIST data
from tensorflow.keras.datasets import mnist

(x_train, labels_train), (x_test, labels_test) = mnist.load_data()
x_test = x_test.astype('float32') / 255
if netType == 'MLP':
    x_test = x_test.reshape(10000, 784)
else:
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# Evaluate
outputs = net.predict(x_test)
labels_predicted = np.argmax(outputs, axis=1)
correct_classified = np.sum(labels_predicted == labels_test)
print('Percentage correctly classified MNIST =', 100 * correct_classified / labels_test.size)


def load_emnist_from_data(archive_path, kind='test'):
    # Load EMNIST images directly from archive path 
    labels_path = os.path.join(archive_path, f'emnist-digits-{kind}-labels-idx1-ubyte')
    images_path = os.path.join(archive_path, f'emnist-digits-{kind}-images-idx3-ubyte')
    
    #print(f"Labels path: {labels_path}")
    
    # reaad the labels (offset)
    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    # Read the images (offset by 16 bytes for the header)
    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)

    return images, labels

print("\n--- Testing on EMNIST (from local archive) ---")

archive_folder = "archive\emnist_source_files"

images_emnist, labels_emnist = load_emnist_from_data(archive_folder, kind='test')

# EMNIST needs normalization & reshape 
images_emnist = images_emnist.astype('float32') / 255.0

# fix orientation, avoid any 90s rotation
images_emnist = np.transpose(images_emnist, (0, 2, 1))

# Reshape for model input 
images_emnist = images_emnist.reshape(images_emnist.shape[0], 28, 28, 1)

emnist_outputs = net.predict(images_emnist)
emnist_predicted = np.argmax(emnist_outputs, axis=1)
emnist_correct = np.sum(emnist_predicted == labels_emnist)
accuracy_emnist = 100 * emnist_correct / len(labels_emnist)

print(f"Number of EMNIST: {len(images_emnist)}")

print(f'Percentage correctly classified EMNIST = {accuracy_emnist:.2f}%')


# Find and plot some misclassified samples
misclassified_indices = np.where(labels_predicted != labels_test)[0]
n_plot = min(len(misclassified_indices), 8)
import matplotlib.pyplot as plt
if n_plot > 0:
    fig, axes = plt.subplots(2, n_plot, figsize=(12, 3))
    fig.subplots_adjust(hspace=0.5)
    
    for i in range(n_plot):
        idx = misclassified_indices[i]
        axes[0, i].imshow(x_test[idx].reshape(28, 28), cmap='gray_r')
        axes[0, i].set_title(f'True: {labels_test[idx]}', y=1.05)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)
        if netType == 'MLP':
            output = net.predict(x_test[idx:idx+1].reshape(1, 784))
        else:
            output = net.predict(x_test[idx:idx+1].reshape(1, 28, 28, 1))
            
            
        output = output[0, 0:]
        axes[1, i].bar(range(10), output)
        axes[1, i].set_xticks(range(10))
        axes[1, i].set_title(f'Pred: {np.argmax(output)}', y=1.05)
        axes[1, i].get_yaxis().set_visible(False)
        if i == 0:
            axes[1, i].get_yaxis().set_visible(True)
            axes[1, 0].set_ylabel('Probability')
            axes[1, 0].set_yticks([0, 0.25, 0.5, 0.75, 1])
        
        fig.subplots_adjust(bottom=0.2)
        fig.text(0.51, 0.05, 'Classes', ha='center', va='center')
    plt.savefig("misclassified.pdf")
else:
    print("No misclassified samples found. No plot generated.")

# Find and plot some misclassified samples from EMNIST 
misclassified_indices = np.where(emnist_predicted != labels_emnist)[0]
n_plot = min(len(misclassified_indices), 8)
import matplotlib.pyplot as plt
if n_plot > 0:
    fig, axes = plt.subplots(2, n_plot, figsize=(12, 3))
    fig.subplots_adjust(hspace=0.5)
    
    for i in range(n_plot):
        idx = misclassified_indices[i]
        axes[0, i].imshow(images_emnist[idx].reshape(28, 28), cmap='gray_r')
        axes[0, i].set_title(f'True: {labels_emnist[idx]}', y=1.05)
        axes[0, i].get_xaxis().set_visible(False)
        axes[0, i].get_yaxis().set_visible(False)
        if netType == 'MLP':
            output = net.predict(images_emnist[idx:idx+1].reshape(1, 784))
        else:
            output = net.predict(images_emnist[idx:idx+1].reshape(1, 28, 28, 1))
            
            
        output = output[0, 0:]
        axes[1, i].bar(range(10), output)
        axes[1, i].set_xticks(range(10))
        axes[1, i].set_title(f'Pred: {np.argmax(output)}', y=1.05)
        axes[1, i].get_yaxis().set_visible(False)
        if i == 0:
            axes[1, i].get_yaxis().set_visible(True)
            axes[1, 0].set_ylabel('Probability')
            axes[1, 0].set_yticks([0, 0.25, 0.5, 0.75, 1])
        
        fig.subplots_adjust(bottom=0.2)
        fig.text(0.51, 0.05, 'Classes', ha='center', va='center')
    plt.savefig("misclassified_EMNIST.pdf")
else:
    print("No misclassified samples found. No plot generated.")