import math
from functools import wraps
from typing import List
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
import matplotlib as mpl
import sklearn.metrics

def set_defaut_device():
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)

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

def constrain_max_norm_(tensor, norm, dim=1):
    norm_ratio = tensor.norm(p=2, dim=dim, keepdim=True) / norm
    norm_coeff = torch.where(norm_ratio > 1, norm_ratio, torch.ones_like(norm_ratio))
    tensor /= norm_coeff
    
def l1_updated_(tensor, lr):
    assert lr >= 0
    
    return tensor.copy_(torch.where(tensor.abs() < lr, torch.zeros_like(tensor), tensor - tensor.sign() * lr))

def image_of_matrix(matrix, max_value=None):
    feature = matrix.detach()
        
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

    return np.vectorize(weight_to_rgb, signature='()->(m)')(feature.cpu().numpy() if feature.is_cuda else feature.cpu().numpy())

def show_matrixes(tensor, max_value=None):
    length = len(tensor)
    plt.figure(figsize=(15 * 2, 15 * math.ceil(length / 2)))
    
    for i in range(length):
        plt.subplot(math.ceil(length / 2), 2 , i + 1)
        
        plt.imshow(image_of_matrix(tensor[i]))
        
def show_one_matrix(matrix, size=(8, 8), max_value=None):
    plt.figure(figsize=size)
    plt.imshow(image_of_matrix(matrix, max_value))
                              
def sparse_init_(matrix, ratio):
    row, column = matrix.size()
    for i in range(row):
        non_zero = int(column * ratio)
        
        filling_position = torch.rand(1, column).to(matrix) > 1 - ratio
        rand_weight = torch.randn(1, column).to(matrix) * (2 / np.sqrt(non_zero))
        
        matrix.data[i] = rand_weight * filling_position.float()
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

def train(net, epoch, optimizer, loss, metrics, train_data_loader, validation_data_loader=None, test_data_loader=None,
          custom_optimize_step=None,
          batch_update_callback=None, epoch_update_callback=None,
          print_results=True):
    """Train the net
    
    loss: (outp: Tensor or [Tensor], target: Tensor or [Tensor], net) -> Tensor
    a metric: (net, data_loader) -> Tensor
    custom_optimize_step has parameter type (net, inpt, target, epoch, iteration)
    other callbacks have parameter types (net, epoch, iteration)
    batch_update_callback can return a tensor

    Return value:
    A tensor of size (category, epoch, metric)
    """
    def print_or_silent(*args):
        if print_results:
            print(*args)

    training_metrics = []
    validation_metrics = []
    test_metrics = []

    batch_metrics = []
    
    iteration_count = 0
    for i in range(epoch):
        print_or_silent('epoch ', i)
        print_or_silent('')
        running_loss = 0.0
        for j, batch in enumerate(train_data_loader):
            inpt, target = batch
                        
            if custom_optimize_step is None:
                optimizer.zero_grad()
                outp = net(inpt)
                
                losses = loss(outp, target, net)
                loss_value = sum(losses) if isinstance(losses, list) else losses
                loss_value.backward()

                optimizer.step()
            else:
                custom_optimize_step(net, inpt, target, i, iteration_count)
                    
            if batch_update_callback is not None:
                t = batch_update_callback(net, i, iteration_count)
                if t is not None:
                    batch_metrics.append(t)
                            
            iteration_count += 1

        if epoch_update_callback is not None:
            epoch_update_callback(net, i, iteration_count)
        
        print_or_silent('training:')
        metric_values = validate(net, train_data_loader, metrics, eval_net=True, print_results=print_results)
        training_metrics.append(metric_values)
        
        if validation_data_loader is not None:
            print_or_silent('')
            print_or_silent('validation:')
            metric_values = validate(net, validation_data_loader, metrics, eval_net=True, print_results=print_results)
            validation_metrics.append(metric_values)

        if test_data_loader is not None:
            print_or_silent('')
            print_or_silent('test:')
            metric_values = validate(net, test_data_loader, metrics, eval_net=True, show_test_mark=True, print_results=print_results)
            test_metrics.append(metric_values)
        
        print_or_silent('--')        

    history = [training_metrics]

    if validation_data_loader is not None:
        history.append(validation_metrics)

    if test_data_loader is not None:
        history.append(test_metrics)

    if len(batch_metrics) > 0:
        return (make_tensor_history(history), torch.stack(batch_metrics))
    else:
        return make_tensor_history(history)

def make_tensor_history(epoch_history):
    def flatten_metrics(metrics: List[torch.Tensor]) -> torch.Tensor:
        """Makes a single Tensor that is the cat of all metrics"""
        return torch.cat([a_metric.view(-1).detach() for a_metric in metrics])

    def stack_between_epochs(one_category_metrics: List[List[torch.Tensor]]):
        return torch.stack([flatten_metrics(metrics) for metrics in one_category_metrics])

    def stack_between_categories(epoch_history):
        return torch.stack([stack_between_epochs(one_category_metrics) for one_category_metrics in epoch_history])

    # Construct a tensor of size (category, epoch, metric),
    # where category means training, validation, test
    return stack_between_categories(epoch_history)

def show_training_history(history):
    def show_one_type(tensor_history):
        """tensor_history has size  (category, epoch, metric)"""
        for metric_index in range(tensor_history.size()[2]):
            epochs = tensor_history.size()[1]

            for category_index in range(tensor_history.size()[0]):
                plt.plot(range(epochs), tensor_history[category_index, :, metric_index].cpu().numpy())
            plt.show()

    if isinstance(history, tuple):
        tensor_epoch_history, tensor_batch_history = history
        # tensor_batch_history is a tensor of size (batch_index, value), add a category dimention
        tensor_batch_history.unsqueeze_(0)

        show_one_type(tensor_epoch_history)
        show_one_type(tensor_batch_history)
    else:
        tensor_epoch_history = history
        show_one_type(tensor_epoch_history)

def validate(net, data_loader, metrics, eval_net=True, show_test_mark=False, print_results=True):
    def print_or_silent(*args):
        if print_results:
            print(*args)

    if eval_net:
        originally_training = net.training
        net.eval()

    if show_test_mark:
        print_or_silent('------>')

    results = []
    for m in metrics:
        loss = m(net, data_loader).detach()
        if loss.numel() == 1:
            print_or_silent(loss.view([]).item())
        else:
            print_or_silent(loss.tolist())
        results.append(loss)

    if eval_net:
        if originally_training:
            net.train()

    return results

def batch_accumulate(batch_average=True):
    """Transform a function of the form of (outp: Tensor or [Tensor], target: Tensor or [Tensor], net) -> metric: Tensor
     to the form of (net, data_loader) -> metric: Tensor"""
    def transform(f):
        @wraps(f)
        def accumulating_f(net, data_loader):
            acc = 0
            count = 0
            for batch in data_loader:
                inpt, target = batch
                batch_size = len(inpt)
                
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

def batch_accuracy(outp, target, net):
    _, predict = outp.max(1)
    _, target_values = target.max(1)

    return (predict == target_values).detach().float().mean()

accuracy = batch_accumulate()(batch_accuracy)

def cohen_kappa(net, data_loader):
    predict_array = []
    target_values_array = []
    for batch in data_loader:
        inpt, target = batch

        outp = net(inpt)
        _, predict = outp.detach().max(1)
        _, target_values = target.detach().max(1)

        predict_array.append(predict)
        target_values_array.append(target_values)

    all_predicts = torch.cat(predict_array)
    all_targets = torch.cat(target_values_array)

    kappa = sklearn.metrics.cohen_kappa_score(all_predicts.cpu().numpy(), all_targets.cpu().numpy())
    return torch.tensor(kappa)

# Use this if you don't need any metric
def zero_metric(net, data_loader):
    return torch.tensor(0.0)

def sensitivity_specificity(net, data_loader):
    subgroups = [[1], [0]] # here assume the 0 label is negative, 1 label is positive
    results = []
    for group in subgroups:
        correct_count = 0
        total_count = 0
        for batch in data_loader:
            inpt, target = batch

            outp = net(inpt)
            _, predict = outp.detach().max(1)
            _, target_values = target.detach().max(1)

            for i in range(inpt.shape[0]):
                if target_values[i] in group:
                    correct_count += 1 if predict[i] == target_values[i] else 0
                    total_count +=1
        results.append(correct_count / total_count)

    return torch.tensor(results)

def simple_loss(tensor_loss):
    """Transform  a function of the form of (outp: Tensor, target: Tensor) -> loss: Tensor
     to the form of (outp: Tensor, target: Tensor, net) -> loss: Tensor"""
    def needed_loss(outp, target, net):
        return tensor_loss(outp, target)
    return needed_loss


class OutputMapNet(nn.Module):
    def __init__(self, net, func):
        super().__init__()
        self.net = net
        self.func = func

    def forward(self, x):
        return self.func(self.net.forward(x))


def map_metric(metric, output_map=None, target_map=None):
    def new_metric(net, data_loader):
        if output_map is not None:
            mapped_net = OutputMapNet(net, output_map)
        else:
            mapped_net = net

        if target_map is not None:
            mapped_loader = map_iterable(lambda batch: (batch[0], target_map(batch[1])), data_loader)
        else:
            mapped_loader = data_loader
        return metric(mapped_net, mapped_loader)
    return new_metric


def show_roc(net, data_loader, index=1, show=True):
    """Show and returns the AOC, false positive, true positive

    index is the index of positive prediction
    """
    originally_training = net.training
    net.eval()

    predicts = []
    targets = []
    for batch in data_loader:
        inpt, target = batch

        predict = net(inpt)
        
        predicts.append(predict)
        targets.append(target)

    total_predicts = torch.cat(predicts).detach()
    total_targets = torch.cat(targets).detach()

    false_positive, true_positive, _ = sklearn.metrics.roc_curve(total_targets[:, index], total_predicts[:, index])
    if show:
        plt.plot(false_positive, true_positive)

    if originally_training:
        net.train()

    return sklearn.metrics.roc_auc_score(total_targets[:, index], total_predicts[:, index]), false_positive, true_positive

def linear_models(sequence):
    return [model for model in sequence if isinstance(model, nn.Linear)]

class SimpleLoaderIter:
    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.current_index = 0
        
    def __next__(self):
        if self.current_index >= len(self.data):
            raise StopIteration()
        else:
            batch_data = self.data[self.current_index:self.current_index + self.batch_size]
            batch_labels = self.labels[self.current_index:self.current_index + self.batch_size]
                        
            self.current_index += self.batch_size
            return batch_data, batch_labels
        

class SimpleLoader:
    def __init__(self, data, labels, batch_size, noise=0.0, shuffle=True):
        self.data = data
        self.labels = labels            
        self.batch_size = batch_size
        self.noise = noise
        self.shuffle=shuffle
        
    def __iter__(self):
        if self.shuffle:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            
            self.data = self.data[indices]
            self.labels = self.labels[indices]

        if self.noise != 0:
            noise_tensor = torch.randn_like(self.data) * self.noise
        
        return SimpleLoaderIter(self.data + noise_tensor if self.noise !=0 else self.data,
                                 self.labels, self.batch_size)

class IterableOperator:
    def __init__(self, iterator_constructor):
        self.iterator_constructor = iterator_constructor

    def __iter__(self):
        # Create a brand new iterator
        return self.iterator_constructor()

def map_iterable(f, iterable):
    return IterableOperator(lambda : map(f, iterable))

class AverageNet(nn.Module):
    def __init__(self, nets):
        super().__init__()
        self.nets = nets
        for i, module in enumerate(nets):
            self.add_module(str(i), module)
        
    def forward(self, x):
        return sum(a_net(x) for a_net in self.nets) / len(self.nets)

class MajorityVote(nn.Module):
    def __init__(self, nets):
        super().__init__()
        self.nets = nets
        for i, module in enumerate(nets):
            self.add_module(str(i), module)
        
    def forward(self, x):
        def predict(outp):
            _, index = outp.max(dim=1)
            return one_hot_encode(index.data, 2)
        
        votes = sum(predict(a_net(x)) for a_net in self.nets) / len(self.nets)
        return votes


