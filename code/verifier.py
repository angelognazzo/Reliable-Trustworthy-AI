import argparse
import csv
import torch
import torch.nn.functional as F
from networks import get_network, get_net_name, NormalizedResnet


DEVICE = 'cpu'
DTYPE = torch.float32

# transform the image pixel values into a tensor
# the values of the images are normalized between 0 and 1
def transform_image(pixel_values, input_dim):
    normalized_pixel_values = torch.tensor([float(p) / 255.0 for p in pixel_values])
    if len(input_dim) > 1:
        input_dim_in_hwc = (input_dim[1], input_dim[2], input_dim[0])
        image_in_hwc = normalized_pixel_values.view(input_dim_in_hwc)
        image_in_chw = image_in_hwc.permute(2, 0, 1)
        image = image_in_chw
    else:
        image = normalized_pixel_values

    assert (image >= 0).all()
    assert (image <= 1).all()
    return image

# spec: the path to the test case
# dataset: a string representing the dataset the network was trained on
def get_spec(spec, dataset):
    # set the input dimensions based on the dataset
    input_dim = [1, 28, 28] if dataset == 'mnist' else [3, 32, 32]
    eps = float(spec[:-4].split('/')[-1].split('_')[-1])
    # open the test case file
    test_file = open(spec, "r")
    # read the header of the file 
    test_instances = csv.reader(test_file, delimiter=",")
    # read the reast of the file line by line extracting the input (image pixel values) and the true label
    for i, (label, *pixel_values) in enumerate(test_instances):
        # transform the image pixel values into a tensor
        inputs = transform_image(pixel_values, input_dim)
        # send the tensor to the device
        inputs = inputs.to(DEVICE).to(dtype=DTYPE)
        # trasnform the true label from an String to an integer
        true_label = int(label)
    # returns a new tensor with a dimension of size one inserted at the specified position
    inputs = inputs.unsqueeze(0)
    return inputs, true_label, eps


# load the trained network from the 'nets' folder
def get_net(net, net_name):
    net = get_network(DEVICE, net)
    state_dict = torch.load('../nets/%s' % net_name, map_location=torch.device(DEVICE))
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]
    net.load_state_dict(state_dict)
    net = net.to(dtype=DTYPE)
    net.eval()
    if 'resnet' in net_name:
        net = NormalizedResnet(DEVICE, net)
    return net

# return the lower and upper bounds of the infinity norm of the input image
def infinty_norm(inputs, pert):
    
    # lower = torch.clamp(inputs - pert, min=0.0).to(DEVICE)
    # upper = torch.clamp(inputs + pert, max=1.0).to(DEVICE)
    lower = torch.maximum(inputs - pert, torch.tensor(0))
    upper = torch.minimum(inputs + pert, torch.tensor(1))

    return torch.flatten(lower), torch.flatten(upper)

def neuron(weights, bias, lower, upper):
    mask = weights < 0
    lower[mask], upper[mask] = -1 * upper[mask], -1 * lower[mask]
    
    return torch.dot(lower, weights) + bias, torch.dot(upper, weights) + bias, bias_lower, bias_upper


def relu_relaxation(lower, upper, constraint_lower, constraint_upper):
    
    if lower > 0 and upper > 0:
        pass
    elif lower < 0 and upper < 0:
        lower = 0
        upper = 0
    else:
        # TODO: implement 
        slope = upper / (upper - lower)
        # constraint_upper = 
    

        contraint_lower = 0


def layer(l, lower_weights, upper_weights, lower_bias, upper_bias):
    if type(l) == torch.nn.modules.linear.Linear:
        weights = l.weight
        bias = l.bias
        input_dim = l.in_features
        output_dim = l.out_features
        
        l_list = []
        u_list = []
        for i in range(output_dim):
            l, u = neuron(weights[i], bias[i], lower_weights, upper_weights)
            l_list.append(l)
            u_list.append(u)
        
        # print(torch.tensor(l_list).shape, torch.tensor(u_list).shape)
        return torch.tensor(l_list), torch.tensor(u_list)
    else:
        output_dim = l.out_features
        l_list = []
        u_list = []
        for i in range(output_dim):
            l, u = relu_relaxation(lower_weights, upper_weights, lower_bias, upper_bias)
            l_list.append(l)
            u_list.append(u)
        return 

# net: the actual network object
# inputs: the input image specified in the test case
# eps: the epsilon value specified in the test case
# true_label: the true label of the input image specified in the test case
# return: 1 if the network is verified, 0 otherwise
# TODO: Implement the analysis function
def analyze(net, inputs, eps, true_label):
    lower, upper = infinty_norm(inputs, eps)
    n_layers = len(net.layers)
    for i in range(2, n_layers):
        l = net.layers[i]
        lower, upper = layer(l, lower, upper)
    
    lower = lower.tolist()
    upper = upper.tolist()
    tmp = [lower[true_label] - u for u in upper]
    return min(tmp) > 0
        


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
    parser.add_argument('--net', type=str, required=True, help='Neural network architecture to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    net_name = get_net_name(args.net)
    dataset = 'mnist' if 'mnist' in net_name else 'cifar10'
    
    # args.spec is the path to the test case
    # dataset is a string representing the dataset the network was trained on
    # get the input image, the true label and the epsilon value from the test case
    inputs, true_label, eps = get_spec(args.spec, dataset)
    # get the actual network object. The netowrk is loaded from the 'nets' folder already trained
    net = get_net(args.net, net_name)

    # get the output of the network on the input image
    outs = net(inputs)
    # get the actual prediction by taking the index of the maximum value in the output tensor
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    # call our analysis function
    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
