import random
import itertools
from typing import List, Tuple

import torch


def label_numbers_in_separation(separation, ordered_label_numbers) -> List[List[int]]:
    """Result[i, j] is how many items of label j there are in separation i"""
    assert sum(separation) == sum(ordered_label_numbers)
    
    total = sum(ordered_label_numbers)
    label_ratios = [label_number / total for label_number in ordered_label_numbers]

    float_sample_number_matrix = torch.tensor([[ratio * separation_number for ratio in label_ratios]
                                               for separation_number in separation])

    # After modyfication, this is the result.
    # The 1st dimension is the separation numbers, the 2nd dimension is labels
    sample_number_matrix = float_sample_number_matrix.floor().long()

    residuals = float_sample_number_matrix - sample_number_matrix.float()

    residual_and_index = [(residuals[i, j].item(), i, j) 
        for i in range(residuals.size()[0]) for j in range(residuals.size()[1])]

    residual_and_index.sort(key=lambda t: t[0], reverse=True)

    left_labels = [ordered_label_numbers[j] - sample_number_matrix[:, j].sum().item()
                   for j in range(len(ordered_label_numbers))]

    left_separation_space = [separation[i] - sample_number_matrix[i, :].sum().item()
                             for i in range(len(separation))]

    total_left = sum(left_separation_space)

    for _, i, j in residual_and_index:
        if total_left == 0:
            break
            
        if left_separation_space[i] > 0 and left_labels[j] > 0:
            sample_number_matrix[i, j] += 1
            left_separation_space[i] -= 1
            left_labels[j] -= 1

            total_left -=1

    return sample_number_matrix


def count_labels(labels: torch.Tensor) -> Tuple[List[int], List[int], List[List[int]]]:
    """
    Return values:
        ordered_label_types: the label values for example [0, 1] for binary classification
        ordered_label_numbers: how many number of samples each label has
        indices_groups_by_label: which indices each label has
    """
    _, numbers = labels.max(dim=1)
    label_types = set(numbers.tolist())

    ordered_label_types = sorted(list(label_types))

    # indices_groups_by_label[i] is the indices whose labels are ordered_label_types[i]
    indices_groups_by_label = [[i for i in range(0, len(labels)) if numbers[i] == label]
                               for label in ordered_label_types]

    ordered_label_numbers = [len(group) for group in indices_groups_by_label]
    
    return ordered_label_types, ordered_label_numbers, indices_groups_by_label


def balanced_index_separation(labels, separation):
    """Separate the labels

    separation is the array of numbers of the size of each data set.
    For example [1560, 300, 200] means you want 3 subsets containing 1560, 300, 200 samples respectively.
    The sum of separation should be the same as the sample size of the labels.
    """
    ordered_label_types, ordered_label_numbers, indices_groups_by_label = count_labels(labels)
    label_numbers_in_sep: List[List[int]] = label_numbers_in_separation(separation, ordered_label_numbers)
    
    label_range_starts_in_sep = torch.zeros_like(label_numbers_in_sep)
    label_range_stops_in_sep = torch.zeros_like(label_numbers_in_sep)
    
    for i in range(label_numbers_in_sep.size()[0]):
        if i > 0:
            label_range_starts_in_sep[i] = label_numbers_in_sep[0:i].sum(dim=0)
        label_range_stops_in_sep[i] = label_numbers_in_sep[0:i + 1].sum(dim=0)

    for group_indices in indices_groups_by_label:
        random.shuffle(group_indices)

    indices_in_sep = [[indices_groups_by_label[j][label_range_starts_in_sep[i, j]:label_range_stops_in_sep[i, j]]
                       for j in range(len(ordered_label_types))]
                      for i in range(len(separation))]
    
    indices = [sum(indices_in_sep[i], []) for i in range(len(separation))]

    for subindices in indices:
        random.shuffle(subindices)

    return indices

def split_by_ratios(total_number, ratios):
    assert abs(sum(ratios) - 1) < 1e-5

    accumulated_ratios = list(itertools.accumulate(ratios))
    rounded = [int(round(ratio * total_number)) for ratio in accumulated_ratios]

    return [rounded[i] - (rounded[i - 1] if i >= 1 else 0) for i in range(len(rounded))]


def repeat_indices(original, destination_number):
    repeat, remainder = divmod(destination_number, len(original))
    
    return original * repeat + random.sample(original, remainder)


def balanced_sample_indices(data_labels):
    """Note, a common pattern is 
            balanced_indices = balanced_sample_indices(data_labels[training_indices])
            balanced_training_data = data[training_indices][balanced_indices]
            balanced_training_labels = data_labels[training_indices][balanced_indices]
    """
    _, ordered_label_numbers, indices_groups_by_label = count_labels(data_labels)
    
    max_number = max(ordered_label_numbers)
    
    result = sum((repeat_indices(indices, max_number) for indices in indices_groups_by_label), [])
    random.shuffle(result)
    return result
