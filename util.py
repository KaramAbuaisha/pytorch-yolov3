from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

def predict_transform(prediction, height, width, anchors, num_classes, CUDA=True):
    # prediction is B x C x H x W
    batch_size = prediction.size(0)
    stride =  height // prediction.size(2)
    grid_h = prediction.size(2)
    grid_w = prediction.size(3)
    bbox_attrs = 5 + num_classes
    # anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    anchors = torch.FloatTensor(anchors)
    num_anchors = len(anchors)
    
    prediction = prediction.transpose(1,3).contiguous()
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_h, grid_w, num_anchors, bbox_attrs)
    
    prediction[:,:,:,:,2:] = torch.sigmoid(prediction[:,:,:,:,2:])

    for i in range(grid_h):
        prediction[:,i,:,:,2] += i
    for i in range(grid_w):
        prediction[:,:,i,:,3] += i
    
    # a, b = np.meshgrid(np.arange(grid_w), np.arange(grid_h))

    # x_offset = torch.FloatTensor(a).view(-1,1)
    # y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        anchors = anchors.cuda()
        # x_offset = x_offset.cuda()
        # y_offset = y_offset.cuda()

    # x_y_offset = torch.cat((y_offset, x_offset), 1).repeat(1,num_anchors).view(1, grid_h, grid_w, num_anchors, 2)

    # prediction[:,:,:,:,2:4] += x_y_offset
    prediction[:,:,:,:,2:4] *= stride

    anchors = anchors.repeat(grid_h*grid_w, 1).view(1, grid_h, grid_w, num_anchors, 2)
    prediction[:,:,:,:,:2] = torch.exp(prediction[:,:,:,:,:2])*anchors

    prediction = prediction.view(batch_size, grid_h*grid_w*num_anchors, bbox_attrs)

    return prediction

def write_results(prediction, confidence, num_classes, nms = True, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    
    
    
    box_a = prediction.new(prediction.shape)
    box_a[:,:,0] = (prediction[:,:,2] + prediction[:,:,0]/2) # max x
    box_a[:,:,1] = (prediction[:,:,3] + prediction[:,:,1]/2) # max y
    box_a[:,:,2] = (prediction[:,:,2] - prediction[:,:,0]/2) # min x
    box_a[:,:,3] = (prediction[:,:,3] - prediction[:,:,1]/2) # min
    prediction[:,:,:4] = box_a[:,:,:4]
    

    
    batch_size = prediction.size(0)
    
    output = prediction.new(1, prediction.size(2) + 1)
    write = False


    for ind in range(batch_size):
        #select the image from the batch
        image_pred = prediction[ind]
        

        
        #Get the class having maximum score, and the index of that class
        #Get rid of num_classes softmax scores 
        #Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        

        
        #Get rid of the zero entries
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))

        
        image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        
        #Get the various classes detected in the image
        try:
            img_classes = unique(image_pred_[:,-1])
        except:
             continue
        #WE will do NMS classwise
        for cls in img_classes:
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            

            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

		
        
             #sort the detections such that the entry with the maximum objectness
             #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)
            
            #if nms has to be done
            if nms:
                #For each detection
                for i in range(idx):
                    #Get the IOUs of all boxes that come after the one we are looking at 
                    #in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                    except ValueError:
                        break
        
                    except IndexError:
                        break
                    
                    #Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask       
                    
                    #Remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                    
                    

            #Concatenate the batch_id of the image to the detection
            #this helps us identify which image does the detection correspond to 
            #We use a linear straucture to hold ALL the detections from the batch
            #the batch_dim is flattened
            #batch is identified by extra batch column
            
            
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    
    return output

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res