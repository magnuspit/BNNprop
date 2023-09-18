import numpy as np
import tensorflow as tf
import json
import larq as lq
from dnn2bnn.metrics.fidelity import Fidelity
from dnn2bnn.models.model_manager import ModelManager
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten, Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, MaxPool2D
from tensorflow.keras.utils import to_categorical

#Init experiment configuration.
batch_size=64
epoch_size= 100
num_iter= 5

# Model / data parameters
num_classes = 10
input_shape = (32,32,3)

#Load LARQ configuration.
larq_configuration={}

with open("configuration/config_relu.json") as json_data_file:
    larq_configuration = json.load(json_data_file)

print(larq_configuration)

#Load data.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

#Init accumulators.
vgg_acc=np.zeros(num_iter)
vgg_bin_acc=np.zeros(num_iter)
vgg_fidelity=np.zeros(num_iter)

for i in range(0,num_iter):
    print("*******************************************")
    print("ITERATION: " +str(i))

    # Define VGG-like model
    inputs = Input(shape=input_shape)
    x = lq.layers.QuantConv2D(64, 3, 
                              kernel_quantizer="ste_sign", 
                              kernel_constraint="weight_clip", 
                              use_bias=False, padding="same")(inputs)
    x = BatchNormalization(momentum=0.999, scale=False)(x)
    x = lq.layers.QuantConv2D(64, 3, 
                              kernel_quantizer="ste_sign", 
                              kernel_constraint="weight_clip", 
                              use_bias=False, padding="same")(x)
    x = MaxPool2D((2, 2))(x)
    x = BatchNormalization(momentum=0.999, scale=False)(x)
    
    x = lq.layers.QuantConv2D(128, 3, 
                              kernel_quantizer="ste_sign", 
                              kernel_constraint="weight_clip", 
                              use_bias=False, padding="same")(x)
    x = BatchNormalization(momentum=0.999, scale=False)(x)
    x = lq.layers.QuantConv2D(128, 3, 
                              kernel_quantizer="ste_sign", 
                              kernel_constraint="weight_clip", 
                              use_bias=False, padding="same")(x)
    x = MaxPool2D((2, 2))(x)
    x = BatchNormalization(momentum=0.999, scale=False)(x)

    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    output = Dense(num_classes, activation='softmax')(x)
    vgg = Model(inputs=inputs, outputs=output)

    #Train VGG
    vgg.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
    vgg.fit(x_train, y_train, epochs=epoch_size, batch_size=batch_size,verbose=2)  
    print("VGG trained")

    #Create copies.
    mm=ModelManager(original_model=vgg,larq_configuration=larq_configuration)
    vgg_bin=mm.create_larq_model()    
    vgg_bin.fit(x_train, y_train, epochs=epoch_size, batch_size=batch_size,verbose=2)  
    print("VGG bin trained")

    #Save models.
    vgg.save("vgg_cifar10_i"+str(i)+str(larq_configuration["activation_relu"])+".h5")
    vgg_bin.save("vgg_bin_cifar10_i"+str(i)+str(larq_configuration["activation_relu"])+".h5")

    #Test models.
    score = vgg.evaluate(x_test, y_test, verbose=2)
    vgg_acc[i]=score[1]
    print("VGG: Test loss:", score[0])
    print("VGG: Test accuracy:", score[1])

    score = vgg_bin.evaluate(x_test, y_test, verbose=2)
    vgg_bin_acc[i]=score[1]
    print("VGG BIN: Test loss:", score[0])
    print("VGG BIN: Test accuracy:", score[1])

    #Get fidelity.
    fidelity=Fidelity(original=vgg, surrogate=vgg_bin,x=x_test)    
    fidelity_value=fidelity.accuracy()
    print("FIDELITY vgg vs vgg_bin" + str(fidelity_value))
    vgg_fidelity[i]=fidelity_value


#Final results
print("FINAL RESULTS******************************")
print("VGG accuracy (mean,std): (" + str(np.mean(vgg_acc)) + "," + str(np.std(vgg_acc))+")")
print("VGG_BIN accuracy (mean,std):"+ str(np.mean(vgg_bin_acc)) + "," + str(np.std(vgg_bin_acc))+")")
print("FIDELITY vgg vs vgg_bin (mean,std):" + str(np.mean(vgg_fidelity)) + "," + str(np.std(vgg_fidelity))+")")
