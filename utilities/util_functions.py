import os
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision
from torch import nn

def reproducability_config(random_seed :  int = 0) -> None: 
    '''
    Improving reproducability of the experiments by configuring determinability. 
    However, pytorch does not guarantee completely reproducible results
    even when using identical seeds.
    https://pytorch.org/docs/stable/notes/randomness.html
    '''

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker():
    '''Preserving reproducability in dataloaders'''

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def find_classes(dir):
    '''Finds the classes and their corresponding indexing id'''

    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return class_to_idx


def plot_images(test_loader: DataLoader, N_SHOT: int):
    '''Plots the support images in all the classes'''

    (
    example_support_images, example_support_labels,
    example_query_images, example_query_labels,
    example_class_ids,
    ) = next(iter(test_loader))

    _, ax = plt.subplots()
    plt.title("Support Images")
    dummy_val = 0
    classes_list = list(example_support_labels)
    classes_list2 = classes_list.insert(0,dummy_val)
    list_classes = list(find_classes(dir).keys())
    #plt.yticks(np.arange(0, 1.2, step=0.2)) 
    ax.set_yticks(np.arange(6), list_classes) 
    plt.imshow(
        torchvision.utils.make_grid(
            example_support_images, nrow=N_SHOT
        ).permute(1, 2, 0)
    )
    return plt


def compute_protoype_mean(z_support, support_labels):
    '''Computes the prototype using mean of the class support images'''

    n_way = len(torch.unique(support_labels))
    return torch.cat([
        z_support[
            torch.nonzero(support_labels == label)
        ].mean(dim = 0)
        for label in range(n_way)
    ])   


def compute_protoype_median(z_support, support_labels):
    '''Computes the prototype using median of the class support images'''

    n_way = len(torch.unique(support_labels))
    return torch.cat([
        z_support[
            torch.nonzero(support_labels == label)
        ].median(dim = 0)[0]
        for label in range(n_way)
    ])   


def pairwise(z_query, z_proto, device):
    '''Calculates the pairwise distance between two torch tensors'''

    pdist = nn.PairwiseDistance(p=2)
    d1 = []
    for j in range(0, len(z_query)):
        d2 = []
        for i in range(0,len(z_proto)):
            d2.append(pdist(z_query[j], z_proto[i]))
        d1.append(d2)
    return(torch.FloatTensor(d1).to(device))


def cosinesimilarity(z_query, z_proto, device):
    '''
    Calculates the pairwise distance between two torch tensors
    #NEEDS FIX
    '''

    cos1 = nn.CosineSimilarity(dim=0, eps=1e-6)
    d1 = []
    for j in range(0, len(z_query)):
        d2 = []
        for i in range(0,len(z_proto)):
            d2.append(cos1(z_query[j], z_proto[i]))
        d1.append(d2)
    return(torch.FloatTensor(d1).to(device))

import splitfolders

def split_dataset():
    splitfolders.fixed(r"C:\DTU\master_thesis\MaritimeFewShotLearning\data\data_2023\val_2023", output="output/",
    seed=0, fixed=(20), oversample=False, group_prefix=None, move=False)
    
