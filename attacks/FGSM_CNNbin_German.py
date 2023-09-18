from larq.layers import QuantDense
from tensorflow.keras.models import load_model
from cleverhans.tf2.attacks import fast_gradient_method
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Register custom layer classes
custom_objects = {'QuantDense': QuantDense}

# Load the pre-trained binary-quantized MLP model
model = load_model('nameofyourmodel.h5', custom_objects=custom_objects)

# Load and preprocess the German Credit dataset
(ds_train, ds_test), info = tfds.load(
    'german_credit_numeric',
    split=['train[0:85%]', 'train[85%:]'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# Convert TensorFlow Dataset to NumPy arrays
X_test = []
Y_test = []
for x, y in tfds.as_numpy(ds_test):
    X_test.append(x)
    Y_test.append(y)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

# Reshape the data to match the model's expected input and output shapes
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
Y_test = Y_test.reshape(Y_test.shape[0], 1)

X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)  # Convert to TensorFlow tensor

# Define loss function
loss_fn = tf.keras.losses.BinaryCrossentropy()

with tf.GradientTape() as tape:
    tape.watch(X_test_tensor)  # Watch the TensorFlow tensor
    prediction = model(X_test_tensor)
    loss = loss_fn(Y_test, prediction)
    
gradients = tape.gradient(loss, X_test_tensor)

# Generate adversarial examples using FGSM
epsilon = 0.1
X_test_adv = X_test + epsilon * np.sign(gradients.numpy())

# Check if these manually created adversarial examples are different
if np.any(X_test_adv != X_test):
    print("Manually created adversarial examples are different from the original examples.")
else:
    print("Manually created adversarial examples are NOT different from the original examples.")

# Evaluate the model on adversarial examples
Y_test_pred = model.predict(X_test_adv)
Y_test_pred = (Y_test_pred > 0.5).astype(int)  # Threshold at 0.5

# Calculate accuracy
acc = np.sum(Y_test_pred == Y_test) / Y_test.shape[0]
print('Test accuracy on adversarial examples: %0.2f%%' % (acc * 100))