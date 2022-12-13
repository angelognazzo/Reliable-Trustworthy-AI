import argparse
import csv
import torch
import torch.nn.functional as F
from networks import get_network, get_net_name, NormalizedResnet
from DeepPolyNetwork import DeepPolyNetwork
from DeepPolyAlphaLoss import DeepPolyAlphaLoss
from DeepPolyReluLayer import DeepPolyReluLayer
import torch.optim as optim



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


# net: the actual network object
# inputs: the input image specified in the test case
# eps: the epsilon value specified in the test case
# true_label: the true label of the input image specified in the test case
# return: 1 if the network is verified, 0 otherwise
def analyze(net, inputs, eps, true_label):    
    def count_parameters2(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # create a DeepPolyNetwork object
    deepPolyNetwork = DeepPolyNetwork(net, eps)
    # _, lower_bounds_list, upper_bounds_list = deepPolyNetwork(inputs)
    # lower = lower_bounds_list[-1].tolist()[0]
    # upper = upper_bounds_list[-1].tolist()[0]

    # print(true_label)
    # print(lower)
    # print(upper)

    # counter = 0
    # for u in upper:
    #     counter += int((lower[true_label] > u))
    # if counter == 9:
    #     return True
    
    for module in deepPolyNetwork.modules():
         if type(module) != DeepPolyReluLayer:
             module.requires_grad_(False)
         else:
             module.requires_grad_(True)
            
    # def count_parameters(model):
    #     total_params = 0
    #     for name, parameter in model.named_parameters():
    #         if not parameter.requires_grad: continue
    #         params = parameter.numel()
    #         print(f"Layer: {name} | Params: {params}")
    #         total_params += params
    #     return total_params
    # # print(count_parameters2(deepPolyNetwork))
    # # print(count_parameters(deepPolyNetwork))
    
    # create Loss
    deepPolyAlphaLoss = DeepPolyAlphaLoss()
    # create optimizer
    optimizer = optim.Adam(deepPolyNetwork.parameters(), lr=0.1)

    # get the output bounds of the network (last tensor of the list) and bring it to a list
    counter_loss = 1
    boh=0
    while(True): #True
        print(counter_loss)
        counter_loss += 1
        optimizer.zero_grad()
        
        # forward the input image through the network to create the final output bounds
        _, lower_bounds_list, upper_bounds_list = deepPolyNetwork(inputs)
        loss = deepPolyAlphaLoss(lower_bounds_list[-1], upper_bounds_list[-1], true_label)
        loss.backward()
        optimizer.step()
        # restricted optimization problem: alpha in [0, 1] (this solution is not very elegant, but it works)
        with torch.no_grad():
            for param in deepPolyNetwork.parameters():
                if param.requires_grad:
                    param.data.clamp_(0, 1)
            """for name, param in deepPolyNetwork.named_parameters():
                if param.requires_grad:
                    param.data.clamp_(0, 1)"""
        print("DAJEEEEE ALPHAAAAAAAA")
        print(deepPolyNetwork.parameters())
       
        
        for name, param in deepPolyNetwork.named_parameters():
            if param.requires_grad:
                print(name, param.data)

        lower = lower_bounds_list[-1].tolist()[0]
        upper = upper_bounds_list[-1].tolist()[0]
        
        print(true_label)
        print(lower)
        print(upper)
        # check if there is an intersection between the output bounds and the true label bound
        counter = 0
        for u in upper:
            counter += int((lower[true_label] > u))
        if counter == 9:
            return True

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
