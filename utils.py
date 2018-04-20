import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
import matplotlib as mpl
from torch.autograd import Variable

def one_hot_encode(labels, number_cases):
    index = labels.unsqueeze(1)
    one_hot = torch.Tensor(len(labels), number_cases).zero_()
    one_hot.scatter_(1, index, 1)
    return one_hot

def l1loss(net, factor):
    acc_loss = 0
    for param in net.parameters():
        acc_loss += F.l1_loss(param, target=torch.zeros_like(param), size_average=False)

    return factor * acc_loss

def where(condition, x, y):
    float_condition = condition.float()
    return float_condition * x + (1 - float_condition) * y

def constrain_max_norm_(tensor, norm, dim):
    norm_ratio = tensor.norm(p=2, dim=dim, keepdim=True) / norm
    norm_coeff = where(norm_ratio > 1, norm_ratio, 1)
    tensor /= norm_coeff
    
def l1_updated_(tensor, lr):
    assert lr >= 0
    
    return tensor.copy_(where(tensor.abs() < lr, 0, tensor - tensor.sign() * lr))

def image_of_matrix(matrix, max_value=None):
    feature = matrix
        
    maximum = feature.max()
    minimum = feature.min()

    if max_value is None:
        absolute = max(abs(maximum), abs(minimum))
    else:
        assert max_value > 0
        absolute = max_value

    def weight_to_rgb(weight):
        if weight >= 0:
            x = min(1.0, weight / absolute)
            return mpl.colors.hsv_to_rgb(np.array([0.6, x, 1]))
        else:
            x = min(1.0, -weight / absolute)
            return mpl.colors.hsv_to_rgb(np.array([0.9, x, 1]))

    return np.vectorize(weight_to_rgb, signature='()->(m)')(feature.cpu().numpy() if feature.is_cuda else feature.numpy())

def show_matrixes(tensor, max_value=None):
    length = len(tensor)
    plt.figure(figsize=(15 * 2, 15 * math.ceil(length / 2)))
    
    for i in range(length):
        plt.subplot(math.ceil(length / 2), 2 , i + 1)
        
        plt.imshow(image_of_matrix(tensor[i]))
        
def show_one_matrix(matrix, size=(8, 8)):
    plt.figure(figsize=size)
    plt.imshow(image_of_matrix(matrix))
               
def cuda_like(origin, like):
    return origin.cuda() if like.is_cuda else origin
               
def sparse_init_(matrix, ratio):
    row, column = matrix.size()
    for i in range(row):
        non_zero = int(column * ratio)
        
        filling_position = cuda_like(torch.rand(1, column), matrix) > 1 - ratio
        rand_weight = cuda_like(torch.randn(1, column), matrix) * (2 / np.sqrt(non_zero))
        
        matrix[i] = rand_weight * filling_position.float()
    return matrix
               
def mul_info(matrix, bins):
    def range_of_bins(total, bins):
        step = total / bins
        return [(round(i * step), round((i + 1) * step)) for i in range(bins)]

    def entropy(xs, groups):
        counts = [((start <= xs) & (xs < end)).long().sum() for start, end in groups]
        total = sum(counts)
        ps = [c / total for c in counts]

        return sum(-p * np.log2(p) if p > 0 else 0.0 for p in ps)
    
    assert len(matrix) > 0
    
    xs = sorted(list(matrix[:, 0]))
    ys = list(set(matrix[:, 1]))
    
    groups = [(xs[start], (xs[end] if end < len(matrix) else xs[end - 1] + 1) ) for (start, end) in range_of_bins(len(matrix), bins)]
    
    # H(X)
    hx = entropy(matrix[:, 0], groups)
    
    p_y0 = (matrix[:, 1] == ys[0]).float().mean()
    p_y1 = (matrix[:, 1] == ys[1]).float().mean()
    
    hxy0 = entropy(matrix[:, 0].masked_select(matrix[:, 1] == ys[0]), groups)
    hxy1 = entropy(matrix[:, 0].masked_select(matrix[:, 1] == ys[1]), groups)
    
    # H(X|Y)
    hxy = p_y0 * hxy0 + p_y1 * hxy1
    
    return hx - hxy

