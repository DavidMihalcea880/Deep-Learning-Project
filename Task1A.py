# Import the necessary libraries
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# Defining the model architecture
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
# Defininng the loss function 
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#Compiling the model before running training 

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

#fitting the model to the training dataset
model.fit(x_train, y_train, epochs=10, batch_size=128)
#Finding predictions for so loss function for training can be found
predictions_train = model.predict(x_train[:1])
# Running the loss functions for the training data set
train_loss = loss_fn(y_train[:1], predictions_train).numpy()
#Using softmax for the predictions
tf.nn.softmax(predictions_train).numpy()
#Evaluating the model using the test dataset
evaluated_model = model.evaluate(x_test,  y_test, verbose=2)
#Finding the probability of the model
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
#The probability model using the xtest.
probability_model(x_test[:5])  # This one works it out for the first 5 values only.
y_test_pred = model.predict(x_test[:1])
test_loss = loss_fn(y_test[:1], y_test_pred).numpy()
print(predictions_train, train_loss, y_test_pred, test_loss)
print(evaluated_model)

