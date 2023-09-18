from larq.layers import QuantConv2D, QuantDense
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from cleverhans.tf2.attacks import fast_gradient_method

# Registrar estas clases de capas como objetos personalizados
custom_objects = {'QuantConv2D': QuantConv2D, 'QuantDense': QuantDense}

# Load the pre-trained model
model = load_model('nameofyourmodel.h5', custom_objects=custom_objects)

# Load the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_test = X_test.reshape(10000, 28, 28, 1)  # Reshape to (10000, 28, 28, 1)
X_test = X_test.astype('float32')
X_test /= 255
Y_test = to_categorical(Y_test, 10)

# Define the loss function
loss_fn = tf.keras.losses.CategoricalCrossentropy()

# Initialize variables to keep track of overall accuracy
total_correct = 0
total_count = 0

# Batch size
batch_size = 500  # Reduced batch size to moderate computing requirements

# Loop over the test set in batches to generate adversarial examples and evaluate the model
for i in range(0, len(X_test), batch_size):
    X_batch = X_test[i:i+batch_size]
    Y_batch = Y_test[i:i+batch_size]

    # Generate adversarial examples for the current batch
    epsilon = 0.1  # Perturbation rate
    X_batch_adv = fast_gradient_method.fast_gradient_method(model, X_batch, epsilon, np.inf)

    # Evaluate the model on the adversarial examples of the current batch
    Y_batch_pred = np.argmax(model.predict(X_batch_adv), axis=1)
    Y_batch_true = np.argmax(Y_batch, axis=1)

    # Update overall accuracy
    total_correct += np.sum(Y_batch_pred == Y_batch_true)
    total_count += len(Y_batch)

# Calculate final accuracy
acc = total_correct / total_count
print('Test accuracy on adversarial examples: %0.2f%%' % (acc * 100))