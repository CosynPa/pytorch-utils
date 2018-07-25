import random

import torch


def label_numbers_in_separation(separation, ordered_label_numbers):
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


def count_labels(labels):
    _, numbers = labels.max(dim=1)
    label_types = set(numbers.tolist())

    ordered_label_types = sorted(list(label_types))

    # indices_groups_by_label[i] is the indices whose labels are ordered_label_types[i]
    indices_groups_by_label = [[i for i in range(0, len(labels)) if numbers[i] == label]
                               for label in ordered_label_types]

    ordered_label_numbers = [len(group) for group in indices_groups_by_label]
    
    return ordered_label_types, ordered_label_numbers, indices_groups_by_label


def balanced_index_separation(labels, separation):
    ordered_label_types, ordered_label_numbers, indices_groups_by_label = count_labels(labels)
    label_numbers_in_sep = label_numbers_in_separation(separation, ordered_label_numbers)
    
    label_range_starts_in_sep = torch.zeros_like(label_numbers_in_sep)
    label_range_stops_in_sep = torch.zeros_like(label_numbers_in_sep)
    
    for i in range(label_numbers_in_sep.size()[0]):
        if i > 0:
            label_range_starts_in_sep[i] = label_numbers_in_sep[0:i].sum(dim=0)
        label_range_stops_in_sep[i] = label_numbers_in_sep[0:i + 1].sum(dim=0)
        
    indices_in_sep = [[indices_groups_by_label[j][label_range_starts_in_sep[i, j]:label_range_stops_in_sep[i, j]]
                       for j in range(len(ordered_label_types))]
                      for i in range(len(separation))]
    
    indices = [sum(indices_in_sep[i], []) for i in range(len(separation))]
    for i in range(len(indices)):
        random.shuffle(indices[i])
    return indices
