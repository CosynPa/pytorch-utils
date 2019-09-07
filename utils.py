import math
import random
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import List, Callable, Optional, Any, Tuple

import matplotlib as mpl
import numpy as np
import plotly
import plotly.graph_objects as go
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from matplotlib import pyplot as plt


def set_defaut_device():
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)


set_defaut_device()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def one_hot_encode(labels, number_cases):
    index = labels.unsqueeze(1)
    one_hot = torch.zeros(len(labels), number_cases)
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


def heatmap_of_matrix(matrix, max_value=None):
    feature = matrix.detach()

    maximum = feature.max().item()
    minimum = feature.min().item()

    if max_value is None:
        absolute = max(abs(maximum), abs(minimum))
    else:
        assert max_value > 0
        absolute = max_value

    return go.FigureWidget([go.Heatmap(z=feature.cpu().numpy().T, zmax=absolute, zmin=-absolute, colorscale="Picnic")])


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

    return np.vectorize(weight_to_rgb, signature='()->(m)')(
        feature.cpu().numpy() if feature.is_cuda else feature.cpu().numpy()
    )


def show_matrixes(tensor, max_value=None):
    length = len(tensor)
    plt.figure(figsize=(15 * 2, 15 * math.ceil(length / 2)))
    
    for i in range(length):
        plt.subplot(math.ceil(length / 2), 2, i + 1)
        
        plt.imshow(image_of_matrix(tensor[i], max_value))

        
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
    
    groups = [(xs[start], (xs[end] if end < len(matrix) else xs[end - 1] + 1) )
              for (start, end) in range_of_bins(len(matrix), bins)]
    
    # H(X)
    hx = entropy(matrix[:, 0], groups)
    
    p_y0 = (matrix[:, 1] == ys[0]).float().mean()
    p_y1 = (matrix[:, 1] == ys[1]).float().mean()
    
    hxy0 = entropy(matrix[:, 0].masked_select(matrix[:, 1] == ys[0]), groups)
    hxy1 = entropy(matrix[:, 0].masked_select(matrix[:, 1] == ys[1]), groups)
    
    # H(X|Y)
    hxy = p_y0 * hxy0 + p_y1 * hxy1
    
    return hx - hxy


def oscillatory_lr(start, end, period, start_decay, end_decay):
    assert start_decay > 0 and end_decay > 0

    def lr_updater(epoch):
        quotient, remainder = divmod(epoch, period)

        period_start = start / start_decay ** quotient
        period_end = end / end_decay ** quotient

        return period_start + (remainder / period) * (period_end - period_start)

    return lr_updater


def cyclic_lr(max_lr, base_lr, mid, period, max_decay, base_decay):
    assert max_decay > 0 and base_decay > 0

    def lr_updater(epoch):
        quotient, remainder = divmod(epoch, period)

        period_max = max_lr / max_decay ** quotient
        period_base = base_lr / base_decay ** quotient

        if remainder <= mid:
            return period_base + (remainder / mid) * (period_max - period_base)
        else:
            assert period - mid != 0  # This is ensured because mid < remainder < period
            return period_max + (remainder - mid) / (period - mid) * (period_base - period_max)

    return lr_updater


@dataclass
class TrainingHistory:
    """A history tensor has size  (category, epoch, metric)"""

    class Category(Enum):
        TRAINING = "Training"
        VALIDATION = "Validation"
        TEST = "Test"

        def color_dict(self):
            if self == TrainingHistory.Category.TRAINING:
                return dict(color="red")
            elif self == TrainingHistory.Category.VALIDATION:
                return dict(color="blue")
            elif self == TrainingHistory.Category.Test:
                return dict(color="green")
            else:
                return dict()

    training_loss: Optional[torch.Tensor]  # size (epoch or batch)
    epochs: torch.Tensor  # size (category, epoch, metric)
    categories: List[Category]
    batches: Optional[torch.Tensor] = None  # size (1, epoch, metric)


    def show(self, columns=2) -> go.Figure:
        def one_type_traces(tensor_history, categories, showing_legends: set) -> List[List[go.Scatter]]:
            """plotly traces for the tensor history

            Traces for one metric is grouped together.
            tensor_history has size  (category, epoch, metric)
            """

            traces: List[List[go.Scatter]] = []
            for metric_index in range(tensor_history.size()[2]):
                epochs = tensor_history.size()[1]

                traces_for_a_metric: List[go.Scatter] = []

                for category_index, category in zip(range(tensor_history.size()[0]), categories):
                    trace = go.Scatter(x=list(range(epochs)),
                                       y=tensor_history[category_index, :, metric_index].cpu().numpy(),
                                       name=category.value,
                                       line=category.color_dict(),
                                       showlegend=category.value not in showing_legends)
                    traces_for_a_metric.append(trace)
                    showing_legends.add(category.value)

                traces.append(traces_for_a_metric)

            return traces

        showing_legends: set = set()

        traces: List[List[go.Scatter]] = []
        if self.training_loss is not None:
            traces += one_type_traces(self.training_loss[None, :, None],
                                      [TrainingHistory.Category.TRAINING],
                                      showing_legends)

        traces += one_type_traces(self.epochs, self.categories, showing_legends)

        if self.batches is not None:
            traces += one_type_traces(self.batches, [TrainingHistory.Category.TRAINING], showing_legends)

        rows = math.ceil(len(traces) / columns)
        fig = go.Figure(plotly.subplots.make_subplots(rows=rows, cols=columns))
        fig.layout.height = 250 * columns

        current_row = 1
        current_col = 1
        for traces_for_a_metric in traces:
            for trace in traces_for_a_metric:
                fig.add_trace(trace, row=current_row, col=current_col)

            current_col += 1
            if current_col == columns + 1:
                current_col = 1
                current_row += 1

        return fig


@dataclass
class Trainer:
    """Trainer of a net

    loss: (outp: Tensor or [Tensor], target: Tensor or [Tensor], net) -> Tensor
    a metric: (net, data_loader) -> Tensor
    custom_optimize_step has parameter type (net, inpt, target, epoch, iteration)
    other callbacks have parameter types (net, epoch, iteration)
    batch_update_callback can return a tensor
    """

    net: nn.Module
    optimizer: torch.optim.Optimizer
    loss: Callable
    metrics: List[Callable[[nn.Module, torch.utils.data.DataLoader], torch.Tensor]]
    train_data_loader: torch.utils.data.DataLoader
    validation_data_loader: Optional[torch.utils.data.DataLoader] = None
    test_data_loader: Optional[torch.utils.data.DataLoader] = None
    per_batch_training_loss: bool = True
    batch_average_training_loss: bool = True
    validate_training: bool = False
    custom_optimize_step: Optional[Callable[[nn.Module, Any, Any, int, int], None]] = None
    lr_scheduler: Optional[Any] = None  # a learning rate scheduler in torch.optim.lr_scheduler
    batch_update_callback: Optional[Callable[[nn.Module, int, int], Optional[torch.Tensor]]] = None
    epoch_update_callback: Optional[Callable[[nn.Module, int, int], None]] = None
    print_results: bool = True

    history: [TrainingHistory] = field(init=False, default_factory=list)

    def train(self, epoch: int):
        def print_or_silent(*args):
            if self.print_results:
                print(*args)

        training_metrics = []
        validation_metrics = []
        test_metrics = []
        running_losses: Optional[List[float]] = []

        batch_metrics = []

        iteration_count = 0
        for i in range(epoch):
            print_or_silent('epoch ', i)
            print_or_silent('')

            # loss of each batch and the number of samples of each batch
            batch_losses_counts: List[Tuple[float, int]] = []

            for j, batch in enumerate(self.train_data_loader):
                inpt, target = batch
                batch_size = len(inpt)

                if self.custom_optimize_step is None:
                    self.optimizer.zero_grad()
                    outp = self.net(inpt)

                    losses = self.loss(outp, target, self.net)
                    loss_value = sum(losses) if isinstance(losses, list) else losses
                    loss_value.backward()

                    batch_losses_counts.append((loss_value.item(), batch_size))

                    self.optimizer.step()
                else:
                    self.custom_optimize_step(self.net, inpt, target, i, iteration_count)

                if self.batch_update_callback is not None:
                    t = self.batch_update_callback(self.net, i, iteration_count)
                    if t is not None:
                        if len(t.size()) == 0:
                            batch_metrics.append(torch.tensor([t.item()]))
                        else:
                            batch_metrics.append(t)

                iteration_count += 1

            if self.epoch_update_callback is not None:
                self.epoch_update_callback(self.net, i, iteration_count)

            if self.custom_optimize_step is None:
                if self.batch_average_training_loss:
                    total_loss = 0.0
                    total_count = 0
                    for loss, batch_size in batch_losses_counts:
                        total_loss += loss * batch_size
                        total_count += batch_size

                    a_running_loss = total_loss / total_count
                else:
                    a_running_loss = sum(loss for loss, _ in batch_losses_counts)

                print_or_silent('running loss:')
                print_or_silent(a_running_loss)

                if self.per_batch_training_loss:
                    running_losses += [loss for loss, _ in batch_losses_counts]
                else:
                    running_losses.append(a_running_loss)
            else:
                running_losses = None

            if self.validate_training:
                print_or_silent('')
                print_or_silent('training:')
                metric_values = validate(self.net, self.train_data_loader, self.metrics, eval_net=True,
                                         print_results=self.print_results)
                training_metrics.append(metric_values)

            if self.validation_data_loader is not None:
                print_or_silent('')
                print_or_silent('validation:')
                metric_values = validate(self.net, self.validation_data_loader, self.metrics, eval_net=True,
                                         print_results=self.print_results)
                validation_metrics.append(metric_values)

            if self.test_data_loader is not None:
                print_or_silent('')
                print_or_silent('test:')
                metric_values = validate(self.net, self.test_data_loader, self.metrics,
                                         eval_net=True, show_test_mark=True,
                                         print_results=self.print_results)
                test_metrics.append(metric_values)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            print_or_silent('--')

        # history[category][epoch][metric] is a loss tensor
        history: List[List[List[torch.Tensor]]] = []
        categories: List[TrainingHistory.Category] = []

        if self.validate_training:
            history.append(training_metrics)
            categories.append(TrainingHistory.Category.TRAINING)

        if self.validation_data_loader is not None:
            history.append(validation_metrics)
            categories.append(TrainingHistory.Category.VALIDATION)

        if self.test_data_loader is not None:
            history.append(test_metrics)
            categories.append(TrainingHistory.Category.TEST)

        history_obj: TrainingHistory
        if len(batch_metrics) > 0:
            history_obj = TrainingHistory(torch.tensor(running_losses),
                                          make_tensor_history(history),
                                          categories,
                                          torch.stack(batch_metrics).unsqueeze(0))
        else:
            history_obj = TrainingHistory(torch.tensor(running_losses),
                                          make_tensor_history(history),
                                          categories)

        self.history.append(history_obj)


def flatten_metrics(metrics: List[torch.Tensor]) -> torch.Tensor:
    """Makes a single Tensor that is the cat of all metrics"""
    return torch.cat([a_metric.view(-1).detach() for a_metric in metrics])


def make_tensor_history(epoch_history: List[List[List[torch.Tensor]]]):
    def stack_between_epochs(one_category_metrics: List[List[torch.Tensor]]):
        return torch.stack([flatten_metrics(metrics) for metrics in one_category_metrics])

    def stack_between_categories(epoch_history: List[List[List[torch.Tensor]]]):
        return torch.stack([stack_between_epochs(one_category_metrics) for one_category_metrics in epoch_history])

    # Construct a tensor of size (category, epoch, metric),
    # where category means training, validation, test
    return stack_between_categories(epoch_history)


def validate(net, data_loader, metrics, eval_net=True, show_test_mark=False, print_results=True) -> List[torch.Tensor]:
    """Returns a list of tensors. The number of tensors is the same as the number of metrics"""
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


def validate_nets(nets, loaders, metrics) -> torch.Tensor:
    """Validate the nets with the corresponding loaders"""
    assert len(nets) == len(loaders)

    results = []
    for i in range(len(nets)):
        net = nets[i]
        loader = loaders[i]

        results.append(flatten_metrics(validate(net, loader, metrics, print_results=False)))

    return torch.stack(results, dim=0)


def batch_accumulate(batch_average=True):
    """Transform a function of the form of (outp: Tensor or [Tensor], target: Tensor or [Tensor], net) -> metric: Tensor
     to the form of (net, data_loader) -> metric: Tensor.

     This is typically used as a metric. The accumulated result can't be used to calculate the gradient.
     """
    def transform(f):
        @wraps(f)
        def accumulating_f(net, data_loader):
            acc = 0
            count = 0
            for batch in data_loader:
                inpt, target = batch
                batch_size = len(inpt)

                outp = net(inpt)

                metric = f(outp, target, net).detach()
                
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


def whole_accumulated(f):
    """Transform a function of the form of (outp: Tensor or [Tensor], target: Tensor or [Tensor], net) -> metric: Tensor
         to the form of (net, data_loader) -> metric: Tensor"""

    @wraps(f)
    def accumulating_f(net, data_loader):
        outp, targets = get_accumulated_pred_target(net, data_loader)
        return f(outp, targets, net)
    return accumulating_f


def batch_accuracy(outp, target, net=None):
    _, predict = outp.max(1)
    _, target_values = target.max(1)

    return single_batch_accuracy(predict, target_values)


accuracy = whole_accumulated(batch_accuracy)


def single_batch_accuracy(outp, target, net=None):
    return (outp == target).detach().float().mean()


single_accuracy = whole_accumulated(single_batch_accuracy)


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


def count_bool(tensor: torch.ByteTensor) -> float:
    return tensor.float().sum().item()


def single_batch_sensitivity_specificity(predicts, targets, net=None):
    """targets and predicts are arrays of 0's and 1's"""

    positive = count_bool(targets == 1)
    negative = count_bool(targets == 0)
    true_positive = count_bool((targets == 1) & (predicts == 1))
    true_negative = count_bool((targets == 0) & (predicts == 0))

    sensitivity = true_positive / positive if positive > 0 else 1.0
    specificity = true_negative / negative if negative > 0 else 1.0

    return torch.tensor([sensitivity, specificity])


single_sensitivity_specificity = whole_accumulated(single_batch_sensitivity_specificity)


def batch_sensitivity_specificity(predicts, targets, net=None):
    _, predict_values = predicts.detach().max(1)
    _, target_values = targets.detach().max(1)
    return single_batch_sensitivity_specificity(predict_values, target_values)


sensitivity_specificity = whole_accumulated(batch_sensitivity_specificity)


def _sensitivity_specificity_legacy(net, data_loader):
    subgroups = [[1], [0]]  # here assume the 0 label is negative, 1 label is positive
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
                    total_count += 1
        results.append(correct_count / total_count)

    return torch.tensor(results)


def auc_metric(index=1):
    def metric(net, data_loader):
        roc, _, _ = show_roc(net, data_loader, index, show=False)
        return torch.tensor(roc)

    return metric


def NLLLoss(size_average=True):
    """Assume outp is log likelihood"""
    def loss(outp, target):
        x = (-target * outp).sum(dim=1)
        return x.mean(dim=0) if size_average else x.sum(dim=0)
    return loss


def simple_loss(tensor_loss):
    """Transform  a function of the form of (outp: Tensor, target: Tensor) -> loss: Tensor to
     the form of (outp: Tensor, target: Tensor, net) -> loss: Tensor"""
    def needed_loss(outp, target, net):
        return tensor_loss(outp, target)
    return needed_loss


class OutputMapNet(nn.Module):
    def __init__(self, net, func):
        super().__init__()
        self.net = net
        self.func = func

        # Make the training state consistent
        self.train(mode=net.training)

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


def get_accumulated_pred_target(net, data_loader):
    predicts = []
    targets = []
    for batch in data_loader:
        inpt, target = batch

        predict = net(inpt)

        predicts.append(predict)
        targets.append(target)

    total_predicts = torch.cat(predicts).detach()
    total_targets = torch.cat(targets).detach()

    return total_predicts, total_targets


def show_roc(net, data_loader, index=1, show=True):
    """Show and returns the AOC, false positive, true positive

    index is the index of positive prediction
    """

    total_predicts, total_targets = get_accumulated_pred_target(net, data_loader)

    false_positive, true_positive, _ = sklearn.metrics.roc_curve(total_targets[:, index], total_predicts[:, index])
    if show:
        plt.plot(false_positive, true_positive)

    return sklearn.metrics.auc(false_positive, true_positive), false_positive, true_positive


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
        self.shuffle = shuffle
        
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
    return IterableOperator(lambda: map(f, iterable))


@dataclass
class MapDataset(torch.utils.data.Dataset):
    dataset: torch.utils.data.Dataset
    target_map: Callable = lambda x: x
    data_map: Callable = lambda x: x

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, item):
        data, targets = self.dataset.__getitem__(item)
        return self.data_map(data), self.target_map(targets)


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


class DenseLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.cat((self.linear(x), x), dim=1)


class DenseLinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Dropout(dropout),
            DenseLinear(in_features, out_features),
            nn.BatchNorm1d(in_features + out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.seq(x)


def fig_quantile_statistic(weights, names, quantile):
    num_features = weights[0].size()[1]
    assert num_features == len(names), "The number of names should equal to the number of input"
    all_ratio = torch.zeros(len(weights), num_features)
    for i, weight in enumerate(weights):
        abs_weight = weight.data.abs()

        weight_mean = abs_weight.mean()
        weight_std = abs_weight.std()

        ratio = (abs_weight > weight_mean + quantile * weight_std).float().mean(dim=0)
        all_ratio[i] = ratio

    if all_ratio.size()[0] >= 2:
        error = dict(
            type="data",
            array=all_ratio.std(dim=0).tolist()
        )
    else:
        error = None

    return go.FigureWidget([go.Bar(x=all_ratio.mean(dim=0).tolist(), error_x=error, y=names, orientation = 'h')]),\
        all_ratio
