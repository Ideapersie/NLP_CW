import numpy as np 
import os 
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 1. LOAD Data , load as per tutorial
(x_train, labels_train), (x_test, labels_test) = mnist.load_data()

# 2. Preporcessing, normalise pixel values 
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print(f"X_train datasize check: {len(x_train)}")

# Encode categorical labels 
y_train = to_categorical(labels_train, 10)
y_test = to_categorical(labels_test, 10)

# Reshape CNN input: no. of exemplars, width, height, colour channels
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 3. Build Neural Network
net = Sequential()

# Split the network into 3 main blocks
net.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))
net.add(BatchNormalization())
net.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', padding='same'))
net.add(BatchNormalization())
net.add(MaxPool2D(pool_size=(2, 2)))
# Light dropout early on
net.add(Dropout(0.25))

# Block 2 
net.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
net.add(BatchNormalization())
net.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
net.add(BatchNormalization())
net.add(MaxPool2D(pool_size=(2, 2)))
# Light dropout early on
net.add(Dropout(0.25))

# Classification black 
net.add(Flatten())
net.add(Dense(256, activation='relu'))
net.add(BatchNormalization())
net.add(Dropout(rate=0.5)) # Heavy dropout before classification
net.add(Dense(10, activation='softmax'))


# 4. Compile network
net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. Data Augmentation, generalize against raw data + handle slight tilts
datagen = ImageDataGenerator(
    rotation_range=15, # Handling small tilts
    zoom_range=0.1, # Handle scale chagnes
    width_shift_range=0.1, # Handle width shift
    height_shift_range=0.1, 
    shear_range=0.1, # Handle italicized handwriting
    fill_mode="nearest"
)

datagen.fit(x_train)


# 6. Train network 
history = net.fit(datagen.flow(x_train, y_train, batch_size=256), 
                  validation_data=(x_test, y_test), 
                  epochs=20,
                  workers=4,
                  use_multiprocessing=True  
                )

# 7. Save network
net.save("my_NN_3blocks.h5")
print("Model saved as my_NN.h5")

# 8. Evaluate model accuracy 
loss, accuracy = net.evaluate(x_test, y_test)
print(f"Standard MNST Test accuracy: {accuracy*100:.2f}%")



