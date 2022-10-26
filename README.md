# ReliableAI 2022 Course Project

This is the project for Reliable and Trustworthy Artificial Intelligence course at ETH Zurich.

## Folder structure
In the directory `code` you can find several files. 
File `networks.py` and `resnet.py` contain encodings of fully connected, convolutional and residual neural network architectures as PyTorch classes.
The architectures extend `nn.Module` object and consist of standard PyTorch layers (e.g. `Linear`, `Flatten`, `ReLU`, `Conv2d`). Please note that first layer of each network performs normalization of the input image. 
File `verifier.py` contains a template of verifier. Loading of the stored networks and test cases is already implemented in `main` function. If you decide to modify `main` function, please ensure that parsing of the test cases works correctly. Your task is to modify `analyze` function by building upon DeepPoly convex relaxation. Note that provided verifier template is guaranteed to achieve **0** points (by always outputting `not verified`).

In folder `nets` you can find 10 neural networks (3 fully connected, 4 convolutional, and 3 residual). These networks are loaded using PyTorch in `verifier.py`.
You can find architectures of these networks in `networks.py`.
Note that for ResNet we prepend Normalization layer after loading the network (see `get_net` function in `verifier.py`).
Name of each network contains the dataset the network is trained used on, e.g. `net3_cifar10_fc3.pt` is network which receives CIFAR-10 images as inputs.
In folder `examples` you can find 10 subfolders. Each subfolder is associated with one of the 10 networks. In a subfolder corresponding to a network, you can find 2 example test cases for this network. 
As explained in the lecture, these test cases **are not** part of the set of test cases which we will use for the final evaluation, and they are only here for you to develop your verifier. 

## Setup instructions

We recommend you to install Python virtual environment to ensure dependencies are same as the ones we will use for evaluation.
To evaluate your solution, we are going to use Python 3.7.
You can create virtual environment and install the dependencies using the following commands:

```bash
$ virtualenv venv --python=python3.7
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Running the verifier

We will run your verifier from `code` directory using the command:

```bash
$ python verifier.py --net {net} --spec ../examples/{net}/img{test_idx}_{eps}.txt
```

In this command, `{net}` is equal to one of the following values (each representing one of the networks we want to verify): `net1, net2, net3, net4, net5, net6, net7, net8, net9, net10`.
`test_idx` is an integer representing index of the test case, while `eps` is perturbation that verifier should certify in this test case.

To test your verifier, you can run for example:

```bash
$ python verifier.py --net net1 --spec ../examples/net1/img1_0.0500.txt
```

To evaluate the verifier on all networks and sample test cases, we provide the evaluation script.
You can run this script using the following commands:

```bash
chmod +x evaluate
./evaluate ../examples
```
