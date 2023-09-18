# Verifying properties of Black Box Models (DNNs) using BNNs
This repository contains the code and resources for our experiment, where we use the [dnn2bnn library](https://github.com/vikical/xNN) to generate Binary Neural Networks (BNNs) from Deep Neural Networks (DNNs) and check the BNN models performance and robustness for various datasets using different activation functions, such as Leaky ReLU and ELU. Then we validated the hypothesis that CNN-like networks are outperforming MLP-like ones independently on the activation function. Finally, we applied FGSM attacks of different magnitued on the BNN's input samples to check its robustness. 

## dnn2bnn submodule - Leaky ReLU and ELU test

We have included the [dnn2bnn library](https://github.com/vikical/xNN) as a submodule in this repository to facilitate the experiment setup.

To clone this repository along with the submodule, run:

```bash
git clone --recursive https://github.com/magnuspit/BNNprop.git
```
It's very recommendable to follow the submodule instructions related to the environment creation and libraries installation. In case of looking for replicating the results of the research, two steps must be followed for each dataset:
- Copy the `config/` files to the `xNN/configuration/` folder.
- Run the Python scripts you can find in `xNN/dnn_examples/` changing the config file link and "activation" function.

All the trained BNNs can be found in `models/` properly identified and in Tensorflow format.

## CIFAR10's unusual results with MLP
In order to validate the hypothesis that the MLP binarized architecture was the real issue for managing the CIFAR10 dataset, it has been designed a CNN-like network that clearly outperformed any MLP architecture for colour images. Both the script to run it and the trained model can be found in `hypo/`

## BNNs robustness on FGSM attacks

To manage this experiments, it's need to add the Cleverhans library to the existing environment. It can be done like this:

```bash
pip install cleverhans
```
Once you get it, you can extract results for any model of the `models/` folder applying the scripts that can be found in `attacks/` 
