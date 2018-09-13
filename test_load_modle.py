import numpy as np
import scipy.io as sio
import os
from data_layer import load_pretrained_model

data = sio.loadmat('SketchANetModel/model_with_order_info_256.mat')
print(data['net'].dtype)
print(data['net']['layers'][0].shape)
print(data['net']['layers'][0][0][0][0]['filters'].shape)
print(data['net']['layers'][0][0][0][0]['biases'][0][0].shape)

weights, biases = load_pretrained_model('SketchANetModel/model_with_order_info_256.mat')
print(weights.keys(),biases.keys())
