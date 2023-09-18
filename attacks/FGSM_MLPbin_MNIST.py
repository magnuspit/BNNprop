from larq.layers import QuantConv2D, QuantDense
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from cleverhans.tf2.attacks import fast_gradient_method

custom_objects = {'QuantConv2D': QuantConv2D, 'QuantDense': QuantDense}

model = tf.keras.models.load_model('nameof your modelo.h5', custom_objects=custom_objects)

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_test = X_test.reshape(10000, 28, 28, 1)  # Reshaped to (10000, 28, 28, 1)
X_test = X_test.astype('float32')
X_test /= 255
Y_test = to_categorical(Y_test, 10)

loss_fn = tf.keras.losses.CategoricalCrossentropy()


epsilon = 0.1
X_test_adv = fast_gradient_method.fast_gradient_method(model, X_test, epsilon, np.inf)

Y_test_pred = np.argmax(model.predict(X_test_adv), axis=1)
Y_test_true = np.argmax(Y_test, axis=1)
acc = np.sum(Y_test_pred == Y_test_true) / Y_test_true.shape[0]

print('Test accuracy on adversarial examples: %0.2f%%' % (acc * 100))
