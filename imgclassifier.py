# This is a code for an Image Classifier program using Convolution Neural Network and the CIFAR-10 dataset.
# This program was written as a exercice project for week 2 of season of AI program.


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

X_test.shape

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

classes[3]

X_train = X_train / 255
X_test = X_test / 255

def plot_sample(X, y, index):
    plt.figure(figsize = (10,10))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index][0]])
    plt.show()

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax'),
])

cnn.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=10)

cnn.evaluate(X_test, y_test)

y_pred = cnn.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]
y_pred[:5]

y_test[:5]

plot_sample(X_test, y_test, 3)

classes[y_test[3][0]]