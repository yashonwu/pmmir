import json
import numpy as np
import pickle
import torch
import random

def get_data_splits(data_type):
    if data_type=='shoe':
        dataset_root = "datasets/amazon_shoes/"
        u_i_seq = json.load(open(dataset_root + 'shoes_u_i_seq.json', 'r'))
        train_seq, test_seq = split_list(u_i_seq)
    elif data_type=='dress':
        dataset_root = "datasets/amazon_dresses/"
        u_i_seq = json.load(open(dataset_root + 'dresses_u_i_seq.json', 'r'))
        train_seq, test_seq = split_list(u_i_seq)
    return u_i_seq, train_seq, test_seq, test_seq

def get_embeddings(data_type):
    if data_type=="shoe":
        feature_root = 'dataset/amazon_shoes/image_features/'
        fc = np.load(feature_root + 'fc_feature.npz')['feat']
        att = np.load(feature_root + 'att_feature.npz')['feat']
        absolute_feature = pickle.load(open(feature_root + 'clip_embedding.p', 'rb'))['all']
    elif data_type=="dress":
        feature_root = 'dataset/amazon_dresses/image_features/'
        fc = np.load(feature_root + 'fc_feature.npz')['feat']
        att = np.load(feature_root + 'att_feature.npz')['feat']
        absolute_feature = pickle.load(open(feature_root + 'clip_embedding.p', 'rb'))['all']
    return fc, att, absolute_feature

def get_docnos(data_type):
    if data_type=="shoe":
        dataset_root = "dataset/amazon_shoes/"
        docnos = json.load(open(dataset_root + 'shoes_docnos.json', 'r'))
    elif data_type=="dress":
        dataset_root = "dataset/amazon_dresses/"
        docnos = json.load(open(dataset_root + 'dresses_docnos.json', 'r'))
    return docnos

def split_list(my_list, seed=42, proportion=0.8):
    """
    Split a list into two splits 'split_a' and 'split_b' based on a random
    proportion.

    Returns the splits and the sampled indices for each split.
    """
    # Set the random seed
    random.seed(seed)

    # Get the length of the list
    list_length = len(my_list)

    # Calculate the number of elements to sample for split_a and split_b
    split_a_length = int(list_length * proportion)
    split_b_length = list_length - split_a_length

    # Create a list of all indices in the original list
    all_indices = list(range(list_length))

    # Sample the indices for split_a
    split_a_indices = random.sample(all_indices, split_a_length)

    # Remove the sampled indices from the list of all indices
    for index in split_a_indices:
        all_indices.remove(index)

    # The remaining indices are for split_b
    split_b_indices = all_indices

    # Create the splits based on the sampled indices
    split_a = [my_list[i] for i in split_a_indices]
    split_b = [my_list[i] for i in split_b_indices]

    # Return the splits and the sampled indices
    return split_a, split_b

