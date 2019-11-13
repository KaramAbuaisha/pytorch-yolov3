from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from util import * # (EmptyLayer, DetectionLayer, predict_transform)

def get_test_input():
    import cv2
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_

def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n') # store the lines in a list
    lines = [x.strip() for x in lines if len(x) > 0 and x[0] != '#'] # get read of the empty lines, get rid of comments, get rid of fringe whitespaces
    
    blocks = []
    block = {}
    for line in lines:
        if line[0]=='[':
            if block:
                blocks.append(block)
            block = {}
            block['type']=line[1:-1].strip()
        else:
            key, value=line.split('=')
            block[key.rstrip()]=value.lstrip()
    blocks.append(block)
    
    return blocks

def create_modules(blocks):
    net_info = blocks[0]     #Captures the information about the input and pre-processing    
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    outputs_to_cache = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        
        #check the type of block
        #create a new module for the block
        #append to module_list
        if x['type'] == 'convolutional':
            if 'batch_normalize' in x and int(x['batch_normalize']) == 1:
                batch_norm = True
                bias = False
            else:
                batch_norm = False
                bias = True
        
            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            activation = x["activation"]

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # conv
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)
            
            
            # conv layers will either be linear (in which case no activation), or leaky relu
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)

            # batch norm after activation
            if batch_norm:
                    bn = nn.BatchNorm2d(filters)
                    module.add_module("batch_norm_{0}".format(index), bn)
        
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners=False)
            module.add_module("upsample_{}".format(index), upsample)

        elif x['type'] == 'route':
            layers = x["layers"].split(',')
            start = int(layers[0])
            if len(layers) == 2:
                end = int(layers[1])
            else:
                end = 0
            
            # convert positive index to negative index
            if start > 0: 
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()

            module.add_module("route_{0}".format(index), route)
            if end < 0:
                outputs_to_cache.append(index+start)
                outputs_to_cache.append(index+end)
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                outputs_to_cache.append(index+start)
                filters = output_filters[index + start]

        elif x['type'] == 'shortcut':
            
            start = int(x['from'])

            if start > 0: 
                outputs_to_cache.append(start)
            else:
                outputs_to_cache.append(index+start)

            shortcut = EmptyLayer()
            module.add_module("shortcut_{0}".format(index), shortcut)
        
        elif x['type'] == 'yolo':
            mask = [int (i) for i in x['mask'].split(',')]
            anchors = x['anchors'].split(',')
            anchors = [(int(anchors[i]), int(anchors[i+1])) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
        
        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    
    return (net_info, module_list, set(outputs_to_cache))

class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list, self.outputs_to_cache = create_modules(self.blocks)
        self.height = int(self.net_info['height'])
        self.width = int(self.net_info['width'])

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        cached_outputs = {}   #We cache the outputs for the route layer
        detections = None
        for index, module in enumerate(modules):        
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[index](x)

            elif module_type == "route":
                layers = module["layers"].split(',')
                start = int(layers[0])

                if start != -1:
                    if start < 0: 
                        start = index + start
                    x = cached_outputs[start]

                if len(layers) == 2:
                    end = int(layers[1])
                    if end < 0:
                        end = index + end
                    x = torch.cat((x, cached_outputs[end]), 1)

            elif module_type == "shortcut":
                start = int(module['from'])
            
                if start < 0: 
                    start = index + start

                x += cached_outputs[start]
            
            elif module_type == "yolo":
                anchors = self.module_list[index][0].anchors
                # mask = [int (i) for i in module['mask'].split(',')]
                # anchors = module['anchors'].split(',')
                # anchors = [(int(anchors[i]), int(anchors[i+1])) for i in range(0, len(anchors), 2)]
                # anchors = [anchors[i] for i in mask]
                num_classes = int(module['classes'])
                
                x = x.data
                x = predict_transform(x, self.height, self.width, anchors, num_classes, CUDA)
                
                if detections is None:
                    detections = x
                else:
                    detections = torch.cat((detections, x),1)

            if index in self.outputs_to_cache:
                cached_outputs[index] = x
        
        return detections


    def load_weights(self, weightfile):
        
        #Open the weights file
        fp = open(weightfile, "rb")

        #The first 4 values are header information 
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number 
        # 4. IMages seen 
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        
        #The rest of the values are the weights
        # Let's load them up
        weights = np.fromfile(fp, dtype = np.float32)
        
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                
                conv = model[0]
                
                if (batch_normalize):
                    bn = model[2]
                    
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
                    
                    #Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    
                    #Cast the loaded weights into dims of model weights. 
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                
                    #Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                    
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                    
                    
                #Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                
                #Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
                
    def save_weights(self, savedfile, cutoff = 0):
            
        if cutoff <= 0:
            cutoff = len(self.blocks) - 1
        
        fp = open(savedfile, 'wb')
        
        # Attach the header at the top of the file
        self.header[3] = self.seen
        header = self.header

        header = header.numpy()
        header.tofile(fp)
        
        # Now, let us save the weights 
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]["type"]
            
            if (module_type) == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0
                    
                conv = model[0]

                if (batch_normalize):
                    bn = model[1]
                
                    #If the parameters are on GPU, convert them back to CPU
                    #We don't convert the parameter to GPU
                    #Instead. we copy the parameter and then convert it to CPU
                    #This is done as weight are need to be saved during training
                    cpu(bn.bias.data).numpy().tofile(fp)
                    cpu(bn.weight.data).numpy().tofile(fp)
                    cpu(bn.running_mean).numpy().tofile(fp)
                    cpu(bn.running_var).numpy().tofile(fp)
                
            
                else:
                    cpu(conv.bias.data).numpy().tofile(fp)
                
                
                #Let us save the weights for the Convolutional layers
                cpu(conv.weight.data).numpy().tofile(fp)

if __name__=='__main__':
    model = Darknet("cfg/yolov3.cfg")
    inp = get_test_input()
    pred = model(inp, torch.cuda.is_available())
    print (pred)
    print (pred.shape)