import math
from functools import wraps
from typing import List

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

def train(net, epoch, optimizer, loss, metrics, train_data_loader, test_data_loader, validation_data_loader=None,
                          update_callback=None):
    """Train the net
    
    loss: (outp: Variable or [Variable], target: Tensor or [Tensor], net) -> Variable
    a metric: (net, data_loader) -> Variable
    """
    training_metrics = []
    validation_metrics = []
    test_metrics = []
    
    iteration_count = 0
    for i in range(epoch):
        print('epoch ', i)
        print('')
        running_loss = 0.0
        for j, batch in enumerate(train_data_loader):
            inpt, target = batch
            v_inpt = Variable(inpt)
                        
            optimizer.zero_grad()
            outp = net(v_inpt)
            
            losses = loss(outp, target, net)
            loss_value = sum(losses) if isinstance(losses, list) else losses
            loss_value.backward()            
                        
            optimizer.step()
                    
            if update_callback is not None:
                update_callback(net, i, iteration_count)
                            
            iteration_count += 1
        
        print('training:')
        metric_values = validate(net, train_data_loader, metrics, False)
        training_metrics.append(metric_values)
        
        if validation_data_loader is not None:
            print('')
            print('validation:')
            metric_values = validate(net, validation_data_loader, metrics, False)
            validation_metrics.append(metric_values)

        print('')
        print('test:')
        metric_values = validate(net, test_data_loader, metrics, True)
        test_metrics.append(metric_values)
        
        print('--')

    if validation_data_loader is not None:
        history = [training_metrics, validation_metrics, test_metrics]
    else:
        history = [training_metrics, test_metrics]

    return history

def show_training_history(epoch_history):
    def flatten_metrics(metrics: List[Variable]) -> Variable:
        """Makes a single Variable that is the cat of all metrics"""
        return torch.cat([a_metric.view(-1) for a_metric in metrics])

    def stack_between_epochs(one_category_metrics: List[List[Variable]]):
        return torch.stack([flatten_metrics(metrics) for metrics in one_category_metrics])

    def stack_between_categories(epoch_history):
        return torch.stack([stack_between_epochs(one_category_metrics) for one_category_metrics in epoch_history])

    # Construct a tensor of size (category, epoch, metric),
    # where category means training, validation, test
    tensor_epoch_history = stack_between_categories(epoch_history)

    for metric_index in range(tensor_epoch_history.size()[2]):
        epochs = tensor_epoch_history.size()[1]

        for category_index in range(tensor_epoch_history.size()[0]):
            plt.plot(range(epochs), tensor_epoch_history[category_index, :, metric_index].numpy())
        plt.show()

def validate(net, data_loader, metrics, is_test):
    origin_training = net.training

    net.eval()

    if is_test:
        print('------>')

    results = []
    for m in metrics:
        loss = m(net, data_loader).data
        if loss.numel() == 1:
            print(loss.view(1)[0])
        else:
            print(loss)
        results.append(loss)

    if origin_training:
        net.train()

    return results

def batch_accumulate(batch_average=True):
    """Transform a function of the form of (outp: Variable or [Variable], target: Tensor or [Tensor], net) -> metric: Varaible
     to the form of (net, data_loader) -> metric: Varaible"""
    def transform(f):
        @wraps(f)
        def accumulating_f(net, data_loader):
            acc = 0
            count = 0
            for batch in data_loader:
                inpt, target = batch
                batch_size = len(inpt)
                inpt = Variable(inpt)
                
                outp = net(inpt)

                metric = f(outp, target, net)
                
                if batch_average:
                    acc += batch_size * metric
                else:
                    acc += metric

                count += batch_size

            if batch_average:
                if count == 0:
                    return 0
                else:
                    return acc / count
            else:
                return acc
        return accumulating_f
    return transform

@batch_accumulate()
def accuracy(outp, target, net):
    _, predict = outp.max(1)
    _, target_values = Variable(target).max(1)

    return (predict == target_values).float().mean()

def simple_loss(tensor_loss):
    """Transform  a function of the form of (outp: Variable, target: Variable) -> loss: Varaible
     to the form of (outp: Variable, target: Tensor, net) -> loss: Varaible"""
    def needed_loss(outp, target, net):
        return tensor_loss(outp, Variable(target))
    return needed_loss
