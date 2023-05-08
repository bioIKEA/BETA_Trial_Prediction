#import tensorflow as tf
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F

def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x

def weight_variable(shape):
    variable = nn.Parameter(cuda(torch.Tensor(*shape).double()))
    return nn.init.normal_(variable, std = 0.1)

def bias_variable(shape):
    variable = nn.Parameter(cuda(torch.Tensor(*shape).double()))
    return nn.init.constant_(variable, val = 0.1)

def a_layer(x_shape,units):
    W = weight_variable([x_shape[1], units])
    b = bias_variable([units])
    return W, b

def a_cul(x, w, b):
    return F.relu(torch.matmul(x, w) + b).double()

def regr_lay_tmp(x1,x2):
    return F.linear(F.relu(torch.matmul(x1,x2))).double()

def regr_lay(x1, x2, w0, w1, wnew):
    print('x1.shape', x1.shape)
    print('x2.shape', x2.shape)
    print('w0.shape', w0.shape)
    print('w1.shape', w1.shape)
    #print('[x1,x2].shape', [x1,x2].shape)
    #print('F.relu([x1,x2]).shape', F.relu([x1,x2]).shape)
    #return F.linear(F.relu([x1,x2])).double()
    print('wnew.shape', wnew.shape)
    #return F.linear(torch.matmul(torch.matmul(x1, w0), torch.transpose(torch.matmul(x2, w1), 0, 1)), torch.transpose(wnew,0,1)).double()
    #return torch.matmul(torch.matmul(x1, w0), torch.transpose(torch.matmul(x2, w1), 0, 1)).double()
    return F.linear(torch.matmul(torch.matmul(x1, w0), torch.transpose(torch.matmul(x2, w1), 0, 1)), wnew).double()
    #return F.linear(F.relu(torch.matmul(torch.matmul(x1, w0), torch.transpose(torch.matmul(x2, w1), 0, 1))), wnew).double()
    
def bi_cul(x0, x1, w0, w1):
    print('x0.shape', x0.shape)
    print('w0.shape', w0.shape)
    print('torch.matmul(x0, w0).shape', torch.matmul(x0, w0).shape)
    print('x1.shape', x1.shape)
    print('w1.shape', w1.shape)
    print('torch.matmul(x1, w1).shape', torch.matmul(x1, w1).shape)
    print('torch.transpose(torch.matmul(x1, w1), 0, 1)).shape', torch.transpose(torch.matmul(x1, w1), 0, 1).shape)
    print('torch.matmul(torch.matmul(x0, w0), torch.transpose(torch.matmul(x1, w1), 0, 1)).double().shape', torch.matmul(torch.matmul(x0, w0), torch.transpose(torch.matmul(x1, w1), 0, 1)).double().shape)
    #sys.exit()
    return torch.matmul(torch.matmul(x0, w0), 
                            torch.transpose(torch.matmul(x1, w1), 0, 1)).double()

def bi_layer(x0_dim_1,x1_dim_1,sym,dim_pred):
    if sym == False:
        W0p = weight_variable([x0_dim_1,dim_pred])
        W1p = weight_variable([x1_dim_1,dim_pred])
        return W0p, W1p
    else:
        W0p = weight_variable([x0_dim_1,dim_pred])
        return W0p, W0p

def bi_layer_new(x0_dim_1,x1_dim_1):
    W0p = weight_variable([x0_dim_1,x1_dim_1])
    return W0p
                   
