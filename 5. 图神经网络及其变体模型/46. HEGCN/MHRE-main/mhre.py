import sys
import os

import numpy as np
import random

from collections import OrderedDict
import collections
import json
import pickle
import operator
import math
import datetime
from tqdm import tqdm
from recordclass import recordclass
import nltk.tokenize

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.backends.cudnn.deterministic = True


def custom_print(*msg):
    for i in range(0, len(msg)):
        if i == len(msg) - 1:
            print(msg[i])
            logger.write(str(msg[i]) + '\n')
        else:
            print(msg[i], ' ', end='')
            logger.write(str(msg[i]))


# Load pre-trained word vectors from GloVe 300d

def load_word_embedding(embed_file, tr_vocab, te_vocab):
    embed_vocab = OrderedDict()
    embed_matrix = list()
    embed_vocab['<PAD>'] = 0
    embed_matrix.append(np.zeros(word_embed_dim, dtype=np.float32))
    embed_vocab['<UNK>'] = 1
    embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))
    embed_vocab['<SEP>'] = 2
    embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))
    word_idx = 3

    with open(embed_file, "r") as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) < word_embed_dim + 1:
                continue
            word = parts[0]
            if word not in embed_vocab and (word in tr_vocab or word in te_vocab):
                    vec = [np.float32(val) for val in parts[1:]]
                    embed_matrix.append(vec)
                    embed_vocab[word] = word_idx
                    word_idx += 1
    custom_print('Pre-trained embeddings size:', len(embed_vocab))
    missing_token = 0
    for word in tr_vocab:
        if word not in embed_vocab:
            if tr_vocab[word] >= min_word_count:
                embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))
                embed_vocab[word] = word_idx
                word_idx += 1
            else:
                missing_token += 1
    custom_print('unknown tokens:', missing_token)
    return embed_vocab, np.array(embed_matrix, dtype=np.float32)


# Populate the word vocabulary from the data files

def build_vocab(tr_lines, de_lines, te_lines, save_vocab_file_name, embedding_file):
    test_vocab = OrderedDict()
    for line in de_lines + te_lines:
        psg_list = json.loads(line.strip())['psg_list']
        for psg in psg_list:
            words = psg['text'].split(' ')
            for word in words:
                word = word.strip()
                if len(word) > 0:
                    test_vocab[word] = 1
    train_vocab = OrderedDict()
    for line in tr_lines:
        psg_list = json.loads(line.strip())['psg_list']
        for psg in psg_list:
            words = psg['text'].split(' ')
            for word in words:
                word = word.strip()
                if len(word) > 0:
                    if word not in train_vocab:
                        train_vocab[word] = 1
                    else:
                        train_vocab[word] += 1
    custom_print('Train corpus vocab size:', len(train_vocab))
    custom_print('Test corpus vocab size:', len(test_vocab))
    custom_print('Minimum word frequency:', min_word_count)
    embed_vocab, embed_matrix = load_word_embedding(embedding_file, train_vocab, test_vocab)
    output = open(save_vocab_file_name, 'wb')
    pickle.dump(embed_vocab, output)
    output.close()
    custom_print('Final vocab size:', len(embed_vocab))
    custom_print('Final word embedding matrix size:', embed_matrix.shape)
    return embed_vocab, embed_matrix


def load_vocab(vocab_file):
    with open(vocab_file, 'rb') as f:
        embed_vocab = pickle.load(f)
    return embed_vocab


# Assign co-reference index to entity 1, entity 2, and connecting entities

def get_words_coref(psg1_words, psg2_words, psg1_ems, psg2_ems, common_ents, rm):
    psg1_coref = [1 for i in range(len(psg1_words))]
    psg2_coref = [1 for i in range(len(psg2_words))]

    for loc_tup in rm['arg1_locations']:
        for idx in range(loc_tup[0], loc_tup[1] + 1):
            psg1_coref[idx] = 2
    for loc_tup in rm['arg2_locations']:
        for idx in range(loc_tup[0], loc_tup[1] + 1):
            psg2_coref[idx] = 3
    if use_coref:
        coref_idx = 4
        for ent in common_ents:
            locs = psg1_ems[ent]
            for loc in locs:
                for idx in range(loc[0], loc[1] + 1):
                    psg1_coref[idx] = min(coref_idx, max_coref_count - 1)

            locs = psg2_ems[ent]
            for loc in locs:
                for idx in range(loc[0], loc[1] + 1):
                    psg2_coref[idx] = min(coref_idx, max_coref_count - 1)

            # coref_idx += 1

    return psg1_coref, psg2_coref


# Convert the linear distance to a distance bin index

def get_distance_bin(dist):
    if dist in dist_bin_map:
        return dist_bin_map[dist]
    if dist < 1:
        bin_idx = 0
    elif dist < 2:
        bin_idx = 1
    elif dist < 4:
        bin_idx = 2
    elif dist < 8:
        bin_idx = 3
    elif dist < 16:
        bin_idx = 4
    elif dist < 32:
        bin_idx = 5
    elif dist < 64:
        bin_idx = 6
    elif dist < 128:
        bin_idx = 7
    else:
        bin_idx = 8
    bin_idx += 1
    dist_bin_map[dist] = bin_idx
    return bin_idx


# Determine all the paths from entity 1 to entity 2 via connecting entities

def get_paths(psg1_ems, psg2_ems, common_ents, rm):
    path_list = []
    for ent1_loc in rm['arg1_locations']:
        for ent2_loc in rm['arg2_locations']:
            for ent in common_ents:
                p1_locs = psg1_ems[ent]
                p2_locs = psg2_ems[ent]
                for p1_loc in p1_locs:
                    for p2_loc in p2_locs:
                        d1 = int(math.fabs(ent1_loc[0] - p1_loc[0]))
                        d2 = int(math.fabs(ent2_loc[0] - p2_loc[0]))
                        path_list.append((ent1_loc[0], ent1_loc[1], p1_loc[0], p1_loc[1],
                                          p2_loc[0], p2_loc[1], ent2_loc[0], ent2_loc[1],
                                          d1, d2))

    path_list.sort(key=lambda tup: tup[8] + tup[9])
    path_list = path_list[:min(max_path_cnt, len(path_list))]

    e1_path_s = []
    e1_path_e = []
    c_p1_s = []
    c_p1_e = []
    c_p2_s = []
    c_p2_e = []
    e2_path_s = []
    e2_path_e = []
    e1_dist = []
    e2_dist = []

    for tup in path_list:
        e1_path_s.append(tup[0])
        e1_path_e.append(tup[1])
        c_p1_s.append(tup[2])
        c_p1_e.append(tup[3])
        c_p2_s.append(tup[4])
        c_p2_e.append(tup[5])
        e2_path_s.append(tup[6])
        e2_path_e.append(tup[7])
        e1_dist.append(get_distance_bin(tup[8]))
        e2_dist.append(get_distance_bin(tup[9]))

    adj_mat = np.ones((len(e1_path_s) + 2, len(e1_path_s) + 2), np.float32)
    adj_mat[0][1] = 0
    adj_mat[1][0] = 0

    return e1_path_s, e1_path_e, c_p1_s, c_p1_e, c_p2_s, c_p2_e, e2_path_s, e2_path_e, e1_dist, e2_dist, adj_mat


def get_sent_idx(start_idx, sents_dct):
    for idx in sents_dct:
        if sents_dct[idx][0] <= start_idx <= sents_dct[idx][1]:
            return idx
    print('Error in get_sent_idx function.')
    exit(0)


# Build the adjacency matrix for the graphs

def get_ems_loc_and_adj_mat(psg_ems, psg_words, sents_dct):
    ems = []
    for em_key in psg_ems:
        for loc in psg_ems[em_key]:
            ems.append((loc[0], loc[1], get_sent_idx(loc[0], sents_dct)))
    ems.sort(key=lambda tup: tup[0])

    adj_mat = np.zeros((len(ems), len(ems)), np.float32)
    for i in range(len(ems)):
        adj_mat[i][i] = 1
        for j in range(i + 1, len(ems)):
            if emg_type2 and ' '.join(psg_words[ems[i][0]: ems[i][1]+1]) == ' '.join(psg_words[ems[j][0]: ems[j][1]+1]):
                adj_mat[i][j] = 1
            if emg_type1 and ems[i][2] == ems[j][2]:
                adj_mat[i][j] = 1
            if emg_type3 and j == i + 1:
                adj_mat[i][j] = 1
            adj_mat[j][i] = adj_mat[i][j]
    ent_mention_idx_map = OrderedDict()
    ent_sents_map = OrderedDict()
    idx = 0
    for em in ems:
        ent = ' '.join(psg_words[em[0]: em[1] + 1])
        ent = ent.lower()
        if ent not in ent_mention_idx_map:
            ent_mention_idx_map[ent] = [idx]
            ent_sents_map[ent] = [em[2]]
        else:
            ent_mention_idx_map[ent].append(idx)
            ent_sents_map[ent].append(em[2])
        idx += 1
    return ems, ent_mention_idx_map, ent_sents_map, adj_mat


# Normalize the adjacency matrix of the graphs

def get_laplacian(adj_mat):
    mat_size = adj_mat.shape[0]
    inv_diag_mat = np.zeros((mat_size, mat_size), np.float32)
    for i in range(0, mat_size):
        cnt = 0
        for j in range(0, mat_size):
            cnt += adj_mat[i][j]
        if cnt == 0:
            continue
        inv_diag_mat[i][i] = 1 / math.sqrt(cnt)
    laplacian = np.matmul(inv_diag_mat, adj_mat)
    laplacian = np.matmul(laplacian, inv_diag_mat)
    return laplacian


# Store the samples in recordClass

def get_sample(psg1_words, psg2_words, psg1_ems_map, psg2_ems_map, common_ents, rm,
               psg1_ems, psg1_ents_ems_map, psg1_ents_sents_map, psg1_ems_adj_mat,
               psg2_ems, psg2_ents_ems_map, psg2_ents_sents_map, psg2_ems_adj_mat,
               psg1_sents_dct, psg2_sents_dct):
    psg1_coref, psg2_coref = get_words_coref(psg1_words, psg2_words, psg1_ems_map, psg2_ems_map, common_ents, rm)
    paths = get_paths(psg1_ems_map, psg2_ems_map, common_ents, rm)
    path_adj_mat = paths[-1]
    paths = paths[0: -1]

    psg1_ent_chain = []
    for em in psg1_ems:
        ent =  ' '.join(psg1_words[em[0]: em[1]+1])
        if ent not in psg1_ent_chain:
            psg1_ent_chain.append(ent)

    psg2_ent_chain = []
    for em in psg2_ems:
        ent = ' '.join(psg2_words[em[0]: em[1] + 1])
        if ent not in psg2_ent_chain:
            psg2_ent_chain.append(ent)

    psg1_ems_words = []
    for em in psg1_ems:
        psg1_ems_words.append(psg1_sents_dct[em[2]])

    psg2_ems_words = []
    for em in psg2_ems:
        psg2_ems_words.append(psg2_sents_dct[em[2]])

    merged_ents_sents = []
    merged_ents = []

    arg1_ems_list = psg1_ents_ems_map[rm['arg1']]
    merged_ents.append(rm['arg1'])
    merged_arg1_sents = []
    for sent_id in psg1_ents_sents_map[rm['arg1']]:
        merged_arg1_sents.append((1, sent_id))
    merged_ents_sents.append(merged_arg1_sents)

    arg2_ems_list = psg2_ents_ems_map[rm['arg2']]
    merged_ents.append(rm['arg2'])
    merged_arg2_sents = []
    for sent_id in psg2_ents_sents_map[rm['arg2']]:
        merged_arg2_sents.append((1, sent_id))
    merged_ents_sents.append(merged_arg2_sents)

    common_ent_ems_list = []
    for ent in common_ents:
        common_ent_ems_list.append(psg1_ents_ems_map[ent])
        common_ent_ems_list.append(psg2_ents_ems_map[ent])
        merged_ents.append(ent)
        ent_sents = []
        for sent_id in psg1_ents_sents_map[ent]:
            ent_sents.append((1, sent_id))
        for sent_id in psg2_ents_sents_map[ent]:
            ent_sents.append((2, sent_id))
        merged_ents_sents.append(ent_sents)

    only_psg1_ent_ems_list = []
    only_psg1_ents = []
    for ent in psg1_ents_ems_map:
        if ent == rm['arg1'] or ent in common_ents:
            continue
        only_psg1_ent_ems_list.append(psg1_ents_ems_map[ent])
        merged_ents.append(ent)
        only_psg1_ents.append(ent)
        ent_sents = []
        for sent_id in psg1_ents_sents_map[ent]:
            ent_sents.append((1, sent_id))
        merged_ents_sents.append(ent_sents)

    only_psg2_ent_ems_list = []
    only_psg2_ents = []
    for ent in psg2_ents_ems_map:
        if ent == rm['arg2'] or ent in common_ents:
            continue
        only_psg2_ent_ems_list.append(psg2_ents_ems_map[ent])
        merged_ents.append(ent)
        only_psg2_ents.append(ent)
        ent_sents = []
        for sent_id in psg2_ents_sents_map[ent]:
            ent_sents.append((2, sent_id))
        merged_ents_sents.append(ent_sents)

    merged_adj_mat = np.zeros((len(merged_ents), len(merged_ents)), np.float32)
    for i in range(0, len(merged_ents)):
        merged_adj_mat[i][i] = 1
        ent_i = merged_ents[i]
        ent_i_sents_psg1 = []
        ent_i_sents_psg2 = []
        ent_i_psg = 0
        if i == 0:
            ent_i_sents_psg1 = psg1_ents_sents_map[merged_ents[i]]
        elif i == 1:
            ent_i_sents_psg2 = psg2_ents_sents_map[merged_ents[i]]
            ent_i_psg = 1
        elif 2 <= i < 2 + int(len(common_ent_ems_list) / 2):
            ent_i_sents_psg1 = psg1_ents_sents_map[merged_ents[i]]
            ent_i_sents_psg2 = psg2_ents_sents_map[merged_ents[i]]
            ent_i_psg = 2
        elif 2 + int(len(common_ent_ems_list) / 2) <= i < 2 + int(len(common_ent_ems_list) / 2)\
                + len(only_psg1_ent_ems_list):
            ent_i_sents_psg1 = psg1_ents_sents_map[merged_ents[i]]
        else:
            ent_i_sents_psg2 = psg2_ents_sents_map[merged_ents[i]]
            ent_i_psg = 1

        for j in range(i+1, len(merged_ents)):
            ent_j = merged_ents[j]
            ent_j_sents_psg1 = []
            ent_j_sents_psg2 = []
            ent_j_psg = 0
            if j == 1:
                ent_j_sents_psg2 = psg2_ents_sents_map[merged_ents[j]]
                ent_j_psg = 1
            elif 2 <= j < 2 + int(len(common_ent_ems_list) / 2):
                ent_j_sents_psg1 = psg1_ents_sents_map[merged_ents[j]]
                ent_j_sents_psg2 = psg2_ents_sents_map[merged_ents[j]]
                ent_j_psg = 2
            elif 2 + int(len(common_ent_ems_list) / 2) <= j < 2 + int(len(common_ent_ems_list) / 2) \
                    + len(only_psg1_ent_ems_list):
                ent_j_sents_psg1 = psg1_ents_sents_map[merged_ents[j]]
            else:
                ent_j_sents_psg2 = psg2_ents_sents_map[merged_ents[j]]
                ent_j_psg = 1

            # if ent_i_psg == ent_j_psg:
            #     merged_adj_mat[i][j] = 1
            #     merged_adj_mat[j][i] = 1

            if eg_type1 and len(set(ent_i_sents_psg1).intersection(set(ent_j_sents_psg1))) > 0:
                merged_adj_mat[i][j] = 1
                merged_adj_mat[j][i] = 1

            if eg_type1 and len(set(ent_i_sents_psg2).intersection(set(ent_j_sents_psg2))) > 0:
                merged_adj_mat[i][j] = 1
                merged_adj_mat[j][i] = 1
            if eg_type2 and ent_i in psg1_ent_chain and ent_j in psg1_ent_chain:
                ent_i_idx = psg1_ent_chain.index(ent_i)
                ent_j_idx = psg1_ent_chain.index(ent_j)
                if ent_i_idx == ent_j_idx + 1 or ent_j_idx == ent_i_idx + 1:
                    merged_adj_mat[i][j] = 1
                    merged_adj_mat[j][i] = 1
            if eg_type2 and ent_i in psg2_ent_chain and ent_j in psg2_ent_chain:
                ent_i_idx = psg2_ent_chain.index(ent_i)
                ent_j_idx = psg2_ent_chain.index(ent_j)
                if ent_i_idx == ent_j_idx + 1 or ent_j_idx == ent_i_idx + 1:
                    merged_adj_mat[i][j] = 1
                    merged_adj_mat[j][i] = 1

    psg1_ents_ems_list = list()
    psg1_ents = list()
    # psg1_ents_ems_list.append(psg1_ents_ems_map[rm['arg1']])
    # psg1_ents.append(rm['arg1'])
    for ent in psg1_ents_ems_map:
        if ent == rm['arg1']:
            continue
        psg1_ents_ems_list.append(psg1_ents_ems_map[ent])
        psg1_ents.append(ent)

    psg2_ents_ems_list = list()
    psg2_ents = list()
    # psg2_ents_ems_list.append(psg2_ents_ems_map[rm['arg2']])
    # psg2_ents.append(rm['arg2'])
    for ent in psg2_ents_ems_map:
        if ent == rm['arg2']:
            continue
        psg2_ents_ems_list.append(psg2_ents_ems_map[ent])
        psg2_ents.append(ent)

    joint_graph_size = len(psg1_ents) + len(psg2_ents) + 2
    joint_adj_mat = np.zeros((joint_graph_size, joint_graph_size), np.float32)

    arg1_sents = psg1_ents_sents_map[rm['arg1']]
    for j in range(len(psg1_ents)):
        ent_j_sents = psg1_ents_sents_map[psg1_ents[j]]
        if len(set(arg1_sents).intersection(set(ent_j_sents))) > 0:
            joint_adj_mat[0][2 + j] = 1
            joint_adj_mat[2 + j][0] = 1

    arg2_sents = psg2_ents_sents_map[rm['arg2']]
    for j in range(len(psg2_ents)):
        ent_j_sents = psg2_ents_sents_map[psg2_ents[j]]
        if len(set(arg2_sents).intersection(set(ent_j_sents))) > 0:
            joint_adj_mat[1][2 + len(psg1_ents) + j] = 1
            joint_adj_mat[2 + len(psg1_ents) + j][1] = 1

    for i in range(len(psg1_ents)):
        ent_i_sents = psg1_ents_sents_map[psg1_ents[i]]
        for j in range(i+1, len(psg1_ents)):
            ent_j_sents = psg1_ents_sents_map[psg1_ents[j]]
            if len(set(ent_i_sents).intersection(set(ent_j_sents))) > 0:
                joint_adj_mat[2 + i][2 + j] = 1
                joint_adj_mat[2 + j][2 + i] = 1

    for i in range(len(psg2_ents)):
        ent_i_sents = psg2_ents_sents_map[psg2_ents[i]]
        for j in range(i+1, len(psg2_ents)):
            ent_j_sents = psg2_ents_sents_map[psg2_ents[j]]
            if len(set(ent_i_sents).intersection(set(ent_j_sents))) > 0:
                joint_adj_mat[2 + len(psg1_ents) + i][2 + len(psg1_ents) + j] = 1
                joint_adj_mat[2 + len(psg1_ents) + j][2 + len(psg1_ents) + i] = 1

    for ent in common_ents:
        ent = ent.lower()
        if ent not in psg1_ents or ent not in psg2_ents:
            print('Common entity Error')
            exit(0)
        psg1_idx = psg1_ents.index(ent)
        psg2_idx = psg2_ents.index(ent)
        joint_adj_mat[2 + psg1_idx][2 + len(psg1_ents) + psg2_idx] = 1
        joint_adj_mat[2 + len(psg1_ents) + psg2_idx][2 + psg1_idx] = 1

    psg1_sents = []
    for i in range(len(psg1_sents_dct)):
        psg1_sents.append(psg1_sents_dct[i])

    psg1_sents_ents_map = OrderedDict()
    for ent in psg1_ents_sents_map:
        for sent_idx in psg1_ents_sents_map[ent]:
            if sent_idx not in psg1_sents_ents_map:
                psg1_sents_ents_map[sent_idx] = [ent]
            else:
                psg1_sents_ents_map[sent_idx].append(ent)

    psg1_sents_adj_mat = np.zeros((len(psg1_sents), len(psg1_sents)), np.float32)
    for i in range(len(psg1_sents)):
        if i not in psg1_sents_ents_map:
            continue
        sent_i_ents = psg1_sents_ents_map[i]
        for j in range(i + 1, len(psg1_sents)):
            if j == i + 1:
                psg1_sents_adj_mat[i][j] = 1
                psg1_sents_adj_mat[j][i] = 1
            else:
                if j not in psg1_ents_sents_map:
                    continue
                sent_j_ents = psg1_sents_ents_map[j]
                if len(set(sent_i_ents).intersection(set(sent_j_ents))) > 0:
                    psg1_sents_adj_mat[i][j] = 1
                    psg1_sents_adj_mat[j][i] = 1

    psg2_sents = []
    for i in range(len(psg2_sents_dct)):
        psg2_sents.append(psg2_sents_dct[i])

    psg2_sents_ents_map = OrderedDict()
    for ent in psg2_ents_sents_map:
        for sent_idx in psg2_ents_sents_map[ent]:
            if sent_idx not in psg2_sents_ents_map:
                psg2_sents_ents_map[sent_idx] = [ent]
            else:
                psg2_sents_ents_map[sent_idx].append(ent)

    psg2_sents_adj_mat = np.zeros((len(psg2_sents), len(psg2_sents)), np.float32)
    for i in range(len(psg2_sents)):
        if i not in psg2_sents_ents_map:
            continue
        sent_i_ents = psg2_sents_ents_map[i]
        for j in range(i + 1, len(psg2_sents)):
            if j == i + 1:
                psg2_sents_adj_mat[i][j] = 1
                psg2_sents_adj_mat[j][i] = 1
            else:
                if j not in psg2_sents_ents_map:
                    continue
                sent_j_ents = psg2_sents_ents_map[j]
                if len(set(sent_i_ents).intersection(set(sent_j_ents))) > 0:
                    psg2_sents_adj_mat[i][j] = 1
                    psg2_sents_adj_mat[j][i] = 1

    psg1_sents_ents_adj_mat = np.zeros((len(merged_ents), len(psg1_sents)), np.float32)
    for i in range(len(merged_ents)):
        if i == 0:
            ent = rm['arg1']
            for j in range(len(psg1_sents)):
                if j in psg1_ents_sents_map[ent]:
                    psg1_sents_ents_adj_mat[i][j] = 1
        elif i == 1:
            continue
        elif i < 2 + len(common_ents):
            ent = common_ents[i - 2]
            for j in range(len(psg1_sents)):
                if j in psg1_ents_sents_map[ent]:
                    psg1_sents_ents_adj_mat[i][j] = 1
        elif i < 2 + len(common_ents) + len(only_psg1_ents):
            ent = only_psg1_ents[i - 2 - len(common_ents)]
            for j in range(len(psg1_sents)):
                if j in psg1_ents_sents_map[ent]:
                    psg1_sents_ents_adj_mat[i][j] = 1
        else:
            continue

    psg2_sents_ents_adj_mat = np.zeros((len(merged_ents), len(psg2_sents)), np.float32)
    for i in range(len(merged_ents)):
        if i == 0:
            continue
        elif i == 1:
            ent = rm['arg2']
            for j in range(len(psg2_ents)):
                if j in psg2_ents_sents_map[ent]:
                    psg2_sents_ents_adj_mat[i][j] = 1
        elif i < 2 + len(common_ents):
            ent = common_ents[i - 2]
            for j in range(len(psg2_sents)):
                if j in psg2_ents_sents_map[ent]:
                    psg2_sents_ents_adj_mat[i][j] = 1
        elif i < 2 + len(common_ents) + len(only_psg1_ents):
            continue
        else:
            ent = only_psg2_ents[i - 2 - len(only_psg1_ents) - len(common_ents)]
            for j in range(len(psg2_sents)):
                if j in psg2_ents_sents_map[ent]:
                    psg2_sents_ents_adj_mat[i][j] = 1

    sample = QASample(Psg1Words=psg1_words, Psg2Words=psg2_words, Arg1=rm['arg1'], Arg2=rm['arg2'],
                      Arg1Loc=rm['arg1_locations'], Arg2Loc=rm['arg2_locations'], Psg1Coref=psg1_coref,
                      Psg2Coref=psg2_coref, Paths=paths, Psg1EMs=psg1_ems, Psg2EMs=psg2_ems,
                      Psg1EMsAdjMat=get_laplacian(psg1_ems_adj_mat), Psg2EMsAdjMat=get_laplacian(psg2_ems_adj_mat),
                      Psg1Ents=psg1_ents_ems_list, Psg2Ents=psg2_ents_ems_list, JointAdjMat=joint_adj_mat,
                      Arg1EMs=arg1_ems_list, Arg2EMs=arg2_ems_list, OnlyPsg1EntEMs=only_psg1_ent_ems_list,
                      OnlyPsg2EntEMs=only_psg2_ent_ems_list, BothPsgEntEMs=common_ent_ems_list,
                      MergedAdjMat=get_laplacian(merged_adj_mat), Psg1Sents=psg1_sents, Psg2Sents=psg2_sents,
                      Psg1SentsAdjMat=psg1_sents_adj_mat, Psg2SentsAdjMat=psg2_sents_adj_mat,
                      Psg1SentsEntsAdjMat=psg1_sents_ents_adj_mat, Psg2SentsEntsAdjMat=psg2_sents_ents_adj_mat,
                      Psg1EMsWords=psg1_ems_words, Psg2EMsWords=psg2_ems_words, EntsSents=merged_ents_sents,
                      PathAdjMat=get_laplacian(path_adj_mat), Relation=rm['relation'])
    return sample


# Read data files

def get_data(lines, is_training_data=False):
    custom_print('no. of lines:', len(lines))
    out_of_len = 0
    pos = 0
    neg = 0
    out_of_set = 0
    pos_samples = []
    neg_samples = []
    no_common = 0
    line_idx = 1
    for line in lines:
        # print(line_idx)
        psg_list = json.loads(line.strip())['psg_list']
        psg_idx_ent_info_map = OrderedDict()
        for psg in psg_list:
            psg_ems_map = psg['entity_mentions']
            psg_words = psg['text'].split(' ')
            sents = nltk.sent_tokenize(psg['text'])
            sents_dct = OrderedDict()
            start = 0
            for i in range(len(sents)):
                sents_dct[i] = (start, start + len(sents[i].split()) - 1)
                start = sents_dct[i][1] + 1
            psg_ems, psg_ents_ems_map, psg_ents_sents_map, psg_ems_adj_mat = get_ems_loc_and_adj_mat(psg_ems_map,
                                                                                                     psg_words,
                                                                                                     sents_dct)
            psg_idx_ent_info_map[psg['idx']] = (psg_ems, psg_ents_ems_map, psg_ents_sents_map, psg_ems_adj_mat,
                                                psg_words, sents_dct)

        data_instances = json.loads(line.strip())['instances']
        for data in data_instances:
            psg1_ems, psg1_ents_ems_map, psg1_ents_sents_map, psg1_ems_adj_mat, \
            psg1_words, psg1_sents_dct = psg_idx_ent_info_map[data['psg1_idx']]

            psg2_ems, psg2_ents_ems_map, psg2_ents_sents_map, psg2_ems_adj_mat, \
            psg2_words, psg2_sents_dct = psg_idx_ent_info_map[data['psg2_idx']]

            psg1_ems_map = psg_list[int(data['psg1_idx'])]['entity_mentions']
            psg2_ems_map = psg_list[int(data['psg2_idx'])]['entity_mentions']
            common_ents = set(psg1_ems_map.keys()).intersection(set(psg2_ems_map.keys()))
            sorted_common_ents = []
            for em_text in common_ents:
                em_locs = psg1_ems_map[em_text]
                locs = []
                for loc in em_locs:
                    locs.append(loc[0])
                locs.sort()
                sorted_common_ents.append((em_text, locs[0]))
            sorted_common_ents.sort(key=lambda tup: tup[1])

            if is_training_data and (len(psg1_words) > max_sent_len or len(psg2_words) > max_sent_len):
                out_of_len += 1
                continue

            for rm in data['relation_mentions']:
                if rm['relation'] not in rel_name_to_idx:
                    out_of_set += 1
                    continue
                arg2 = rm['arg2']
                if len(arg2) == 1 and ord(arg2) in range(91, 123):
                    continue
                elif arg2 in ["of", "he", "a", "an", "the", "as", "e .", "s .", "a .", '*', ',', '.', '"']:
                    continue

                cur_common_ents = []
                for tup in sorted_common_ents:
                    if tup[0].lower() in [rm['arg1'], rm['arg2']]:
                        continue
                    cur_common_ents.append(tup[0])

                if len(cur_common_ents) == 0:
                    continue

                sample = get_sample(psg1_words, psg2_words, psg1_ems_map, psg2_ems_map, cur_common_ents, rm,
                                    psg1_ems, psg1_ents_ems_map, psg1_ents_sents_map, psg1_ems_adj_mat,
                                    psg2_ems, psg2_ents_ems_map, psg2_ents_sents_map, psg2_ems_adj_mat,
                                    psg1_sents_dct, psg2_sents_dct)

                if rm['relation'] in ignore_rel_list:
                    neg += 1
                    neg_samples.append(sample)
                else:
                    pos += 1
                    pos_samples.append(sample)
        line_idx += 1
        if line_idx % 100 == 0:
            print("Processed up to:", line_idx)

    custom_print('Unknown relations:', out_of_set)
    custom_print('No linking entities:', no_common)
    if is_training_data:
        custom_print('Passage length more than maximum:', out_of_len)
    return pos_samples, neg_samples


# Determine the best threshold and corresponding F1 score

def get_f1_and_threshold(data, preds):
    gt_pos = 0
    for d in data:
        if d.Relation not in ignore_rel_list:
            gt_pos += 1
    res = []
    for i in range(len(data)):
        j = np.argmax(preds[i])
        if rel_idx_to_name[j] in ignore_rel_list:
            continue
        res.append((rel_idx_to_name[j] == data[i].Relation, preds[i][j], rel_idx_to_name[j]))
    if len(res) == 0:
        return 0.0, 0.0
    res.sort(key=lambda tup: tup[1], reverse=True)
    correct = 0
    prec_list = []
    rec_list = []
    for i in range(len(res)):
        correct += res[i][0]
        prec_list.append(float(correct) / (i + 1))
        rec_list.append(float(correct) / gt_pos)

    prec_list = np.asarray(prec_list, dtype='float32')
    rec_list = np.asarray(rec_list, dtype='float32')
    f1_arr = (2 * prec_list * rec_list / (prec_list + rec_list + 1e-20))

    f1 = f1_arr.max()
    f1_pos = f1_arr.argmax()
    th = res[f1_pos][1]
    custom_print('Prec:', round(prec_list[f1_pos], 3))
    custom_print('Rec:', round(rec_list[f1_pos], 3))
    return round(th, 3), round(f1, 3)


# Calculate F1 wrt to a threshold

def get_F1(data, preds, re_th=0.0, out_file=None):
    gt_pos = 0
    pred_pos = 0
    correct_pos = 0
    cnt = 0
    writer = None
    if out_file is not None:
        writer = open(out_file, 'w')
        writer.write('GT\tPrediction\n')
    for i in range(0, len(data)):
        if data[i].Relation not in ignore_rel_list:
            gt_pos += 1

        j = np.argmax(preds[i])
        if rel_idx_to_name[j] not in ignore_rel_list and preds[i][j] >= re_th:
            pred_pos += 1
            if data[i].Relation == rel_idx_to_name[j]:
                correct_pos += 1
            if data[i].Relation not in ignore_rel_list:
                cnt += 1
            if out_file is not None:
                writer.write(data[i].Relation + '\t' + rel_idx_to_name[j] + '\n')
        else:
            if out_file is not None:
                writer.write(data[i].Relation + '\t' + 'None' + '\n')
    if out_file is not None:
        writer.close()
    custom_print('Relation extraction results')
    custom_print(pred_pos, '\t', gt_pos, '\t', correct_pos, '\t', cnt)
    p = float(correct_pos) / (pred_pos + 1e-8)
    r = float(correct_pos) / (gt_pos + 1e-8)
    f1 = (2 * p * r) / (p + r + 1e-8)
    custom_print('Prec.:', round(p, 3), 'Rec.:', round(r, 3), 'F1:', round(f1, 3))
    return round(f1, 3)


def shuffle_data(data):
    data.sort(key=lambda x: len(x.Psg1Words) + len(x.Psg2Words))
    num_batch = int(len(data) / batch_size)
    rand_idx = random.sample(range(num_batch), num_batch)
    new_data = []
    for idx in rand_idx:
        new_data += data[batch_size * idx: batch_size * (idx + 1)]
    if len(new_data) < len(data):
        new_data += data[num_batch * batch_size:]
    return new_data


def get_class_name_idx_map(rel_file):
    name_to_idx_map = collections.OrderedDict()
    idx_to_name_map = collections.OrderedDict()
    reader = open(rel_file)
    lines = reader.readlines()
    reader.close()
    idx = 0
    for line in lines:
        line = line.strip().split('\t')[0]
        line = line.strip()
        name_to_idx_map[line] = idx
        idx_to_name_map[idx] = line
        idx += 1
    return name_to_idx_map, idx_to_name_map


def get_max_len(cur_samples):
    psg1_max_len = -1
    psg2_max_len = -1
    batch_max_path_cnt = 2
    psg1_max_ems_cnt = -1
    psg2_max_ems_cnt = -1
    max_joint_ent_cnt = -1
    psg1_max_sent_cnt = -1
    psg2_max_sent_cnt = -1
    max_arg1_mention_cnt = 2
    max_arg2_mention_cnt = 2
    for sample in cur_samples:
        if len(sample.Psg1Words) > psg1_max_len:
            psg1_max_len = len(sample.Psg1Words)
        if len(sample.Psg2Words) > psg2_max_len:
            psg2_max_len = len(sample.Psg2Words)
        if len(sample.Paths[0]) > batch_max_path_cnt:
            batch_max_path_cnt = len(sample.Paths[0])
        if len(sample.Psg1EMs) > psg1_max_ems_cnt:
            psg1_max_ems_cnt = len(sample.Psg1EMs)
        if len(sample.Psg2EMs) > psg2_max_ems_cnt:
            psg2_max_ems_cnt = len(sample.Psg2EMs)
        # if len(sample.Psg1Ents) + len(sample.Psg2Ents) + 2 > max_joint_ent_cnt:
        #     max_joint_ent_cnt = len(sample.Psg1Ents) + len(sample.Psg2Ents) + 2
        if len(sample.OnlyPsg1EntEMs) + len(sample.OnlyPsg2EntEMs) + len(sample.BothPsgEntEMs) + 2 > max_joint_ent_cnt:
            max_joint_ent_cnt = len(sample.OnlyPsg1EntEMs) + len(sample.OnlyPsg2EntEMs) + len(sample.BothPsgEntEMs) + 2
        if len(sample.Psg1Sents) > psg1_max_sent_cnt:
            psg1_max_sent_cnt = len(sample.Psg1Sents)
        if len(sample.Psg2Sents) > psg2_max_sent_cnt:
            psg2_max_sent_cnt = len(sample.Psg2Sents)
        if len(sample.Arg1Loc) > max_arg1_mention_cnt:
            max_arg1_mention_cnt = len(sample.Arg1Loc)
        if len(sample.Arg2Loc) > max_arg2_mention_cnt:
            max_arg2_mention_cnt = len(sample.Arg2Loc)
    return psg1_max_len, psg2_max_len, batch_max_path_cnt, psg1_max_ems_cnt, psg2_max_ems_cnt, max_joint_ent_cnt, \
           psg1_max_sent_cnt, psg2_max_sent_cnt, max_arg1_mention_cnt, max_arg2_mention_cnt


# Convert words to embedding indexes

def get_words_index_seq(path_words, max_len, add_separator=False):
    path_seq = list()
    path_mask = [0 for i in range(len(path_words))]
    for word in path_words:
        if word in word_vocab:
            path_seq.append(word_vocab[word])
        else:
            path_seq.append(word_vocab['<UNK>'])
    pad_len = max_len - len(path_words)
    if add_separator:
        pad_len -= 1
    path_mask += [1 for i in range(pad_len)]
    path_seq += [word_vocab['<PAD>'] for i in range(pad_len)]
    if add_separator:
        path_seq += [word_vocab['<SEP>']]
        path_mask += [0]
    return path_seq, path_mask


def get_padded_coref_seq(valid_seq, max_len, add_separator=False):
    padded_seq = [seq_id for seq_id in valid_seq]
    pad_len = max_len - len(valid_seq)
    if add_separator:
        pad_len -= 1
    padded_seq += [0 for i in range(pad_len)]
    if add_separator:
        padded_seq += [1]
    return padded_seq


def get_arg_mask(arg_locs, max_len):
    mask = [0 for i in range(len(arg_locs))] + [1 for i in range(max_len - len(arg_locs))]
    return mask


# Prepare a batch for train/test

def get_batch_data(cur_samples, is_training=False):
    """
    Returns the training samples and labels as numpy array
    """
    psg1_max_len, psg2_max_len, batch_max_path_cnt, psg1_max_ems_cnt, psg2_max_ems_cnt, \
    joint_max_ent_cnt, psg1_max_sent, psg2_max_sent, max_arg1_mention, max_arg2_mention = get_max_len(cur_samples)
    psg1_max_len += 1

    psg1_words_list = list()
    psg1_mask_list = list()
    psg2_words_list = list()
    psg2_mask_list = list()
    psg1_coref_list = list()
    psg2_coref_list = list()
    arg1_s_list = []
    arg1_e_list = []
    arg2_s_list = []
    arg2_e_list = []
    path_e1_s = []
    path_e1_e = []
    path_c1_s = []
    path_c1_e = []
    path_c2_s = []
    path_c2_e = []
    path_e2_s = []
    path_e2_e = []
    path_mask = []
    arg1_mask = []
    arg2_mask = []

    ems_path_e1 = []
    ems_path_c1 = []
    ems_path_c2 = []
    ems_path_e2 = []
    ems_path_mask = []

    psg1_ems_start_list = []
    psg1_ems_end_list = []
    psg1_ems_mask_list = []
    psg1_ems_adj_mat_list = []
    psg1_ems_words_mask_list = []

    psg2_ems_start_list = []
    psg2_ems_end_list = []
    psg2_ems_mask_list = []
    psg2_ems_adj_mat_list = []
    psg2_ems_words_mask_list = []

    joint_ent_list = []
    joint_ent_mask_list = []
    joint_ent_adj_mat_list = []
    joint_ent_sent_mask_list = []
    linked_entities_list = []
    linked_entities_mask = []

    rel_labels_list = []
    sample_idx = 0
    for sample in cur_samples:
        words_seq, words_mask = get_words_index_seq(sample.Psg1Words, psg1_max_len, True)
        psg1_words_list.append(words_seq)
        psg1_mask_list.append(words_mask)

        words_seq, words_mask = get_words_index_seq(sample.Psg2Words, psg2_max_len, False)
        psg2_words_list.append(words_seq)
        psg2_mask_list.append(words_mask)

        psg1_coref_list.append(get_padded_coref_seq(sample.Psg1Coref, psg1_max_len, True))
        psg2_coref_list.append(get_padded_coref_seq(sample.Psg2Coref, psg2_max_len, False))

        cur_arg1_s = [sample_idx * psg1_max_len + idx[0] for idx in sample.Arg1Loc] + \
                     [sample_idx * psg1_max_len for i in range(max_arg1_mention - len(sample.Arg1Loc))]
        cur_arg1_e = [sample_idx * psg1_max_len + idx[1] for idx in sample.Arg1Loc] + \
                     [sample_idx * psg1_max_len for i in range(max_arg1_mention - len(sample.Arg1Loc))]

        cur_arg2_s = [sample_idx * psg2_max_len + idx[0] for idx in sample.Arg2Loc] + \
                     [sample_idx * psg2_max_len for i in range(max_arg2_mention - len(sample.Arg2Loc))]
        cur_arg2_e = [sample_idx * psg2_max_len + idx[1] for idx in sample.Arg2Loc] + \
                     [sample_idx * psg2_max_len for i in range(max_arg2_mention - len(sample.Arg2Loc))]

        arg1_s_list.append(cur_arg1_s)
        arg1_e_list.append(cur_arg1_e)
        arg2_s_list.append(cur_arg2_s)
        arg2_e_list.append(cur_arg2_e)

        arg1_mask.append(get_arg_mask(sample.Arg1Loc, max_arg1_mention))
        arg2_mask.append(get_arg_mask(sample.Arg2Loc, max_arg2_mention))

        path_mask_len = batch_max_path_cnt - len(sample.Paths[0])
        cur_path_mask = [0 for i in range(len(sample.Paths[0]))] + [1 for i in range(path_mask_len)]
        path_mask.append(cur_path_mask)

        cur_path_e1_s = [sample_idx * psg1_max_len + idx for idx in sample.Paths[0]] + \
                        [sample_idx * psg1_max_len for i in range(path_mask_len)]
        cur_path_e1_e = [sample_idx * psg1_max_len + idx for idx in sample.Paths[1]] + \
                        [sample_idx * psg1_max_len for i in range(path_mask_len)]

        cur_path_c1_s = [sample_idx * psg1_max_len + idx for idx in sample.Paths[2]] + \
                        [sample_idx * psg1_max_len for i in range(path_mask_len)]
        cur_path_c1_e = [sample_idx * psg1_max_len + idx for idx in sample.Paths[3]] + \
                        [sample_idx * psg1_max_len for i in range(path_mask_len)]

        cur_path_c2_s = [sample_idx * psg2_max_len + idx for idx in sample.Paths[4]] + \
                        [sample_idx * psg2_max_len for i in range(path_mask_len)]
        cur_path_c2_e = [sample_idx * psg2_max_len + idx for idx in sample.Paths[5]] + \
                        [sample_idx * psg2_max_len for i in range(path_mask_len)]

        cur_path_e2_s = [sample_idx * psg2_max_len + idx for idx in sample.Paths[6]] + \
                        [sample_idx * psg2_max_len for i in range(path_mask_len)]
        cur_path_e2_e = [sample_idx * psg2_max_len + idx for idx in sample.Paths[7]] + \
                        [sample_idx * psg2_max_len for i in range(path_mask_len)]

        path_e1_s.append(cur_path_e1_s)
        path_e1_e.append(cur_path_e1_e)

        path_c1_s.append(cur_path_c1_s)
        path_c1_e.append(cur_path_c1_e)

        path_c2_s.append(cur_path_c2_s)
        path_c2_e.append(cur_path_c2_e)

        path_e2_s.append(cur_path_e2_s)
        path_e2_e.append(cur_path_e2_e)

        cur_psg1_ems_s = [sample_idx * psg1_max_len + tup[0] for tup in sample.Psg1EMs] + \
                         [sample_idx * psg1_max_len for i in range(psg1_max_ems_cnt - len(sample.Psg1EMs))]
        cur_psg1_ems_e = [sample_idx * psg1_max_len + tup[1] for tup in sample.Psg1EMs] + \
                         [sample_idx * psg1_max_len for i in range(psg1_max_ems_cnt - len(sample.Psg1EMs))]
        cur_psg1_ems_mask = [0 for i in range(len(sample.Psg1EMs))] + \
                            [1 for i in range(psg1_max_ems_cnt - len(sample.Psg1EMs))]
        psg1_ems_start_list.append(cur_psg1_ems_s)
        psg1_ems_end_list.append(cur_psg1_ems_e)
        psg1_ems_mask_list.append(cur_psg1_ems_mask)

        cur_psg1_words_mask = []
        for tup in sample.Psg1EMsWords:
            ems_words = [1 for idx in range(psg1_max_len)]
            ems_words[tup[0]: tup[1] + 1] = [0 for i in range(tup[1] + 1 - tup[0])]
            cur_psg1_words_mask.append(ems_words)
        for i in range(psg1_max_ems_cnt - len(sample.Psg1EMs)):
            cur_psg1_words_mask.append([0 for idx in range(psg1_max_len)])
        psg1_ems_words_mask_list.append(cur_psg1_words_mask)

        cur_psg2_ems_s = [sample_idx * psg2_max_len + tup[0] for tup in sample.Psg2EMs] + \
                         [sample_idx * psg2_max_len for i in range(psg2_max_ems_cnt - len(sample.Psg2EMs))]
        cur_psg2_ems_e = [sample_idx * psg2_max_len + tup[1] for tup in sample.Psg2EMs] + \
                         [sample_idx * psg2_max_len for i in range(psg2_max_ems_cnt - len(sample.Psg2EMs))]
        cur_psg2_ems_mask = [0 for i in range(len(sample.Psg2EMs))] + \
                            [1 for i in range(psg2_max_ems_cnt - len(sample.Psg2EMs))]
        psg2_ems_start_list.append(cur_psg2_ems_s)
        psg2_ems_end_list.append(cur_psg2_ems_e)
        psg2_ems_mask_list.append(cur_psg2_ems_mask)

        cur_psg2_words_mask = []
        for tup in sample.Psg2EMsWords:
            ems_words = [1 for idx in range(psg2_max_len)]
            ems_words[tup[0]: tup[1] + 1] = [0 for i in range(tup[1] + 1 - tup[0])]
            cur_psg2_words_mask.append(ems_words)
        for i in range(psg2_max_ems_cnt - len(sample.Psg2EMs)):
            cur_psg2_words_mask.append([0 for idx in range(psg2_max_len)])
        psg2_ems_words_mask_list.append(cur_psg2_words_mask)

        cur_psg1_ems_adj_mat = np.zeros((psg1_max_ems_cnt, psg1_max_ems_cnt), dtype=np.float32)
        cur_psg1_ems_adj_mat[:len(sample.Psg1EMs), :len(sample.Psg1EMs)] = sample.Psg1EMsAdjMat
        psg1_ems_adj_mat_list.append(cur_psg1_ems_adj_mat)

        cur_psg2_ems_adj_mat = np.zeros((psg2_max_ems_cnt, psg2_max_ems_cnt), dtype=np.float32)
        cur_psg2_ems_adj_mat[:len(sample.Psg2EMs), :len(sample.Psg2EMs)] = sample.Psg2EMsAdjMat
        psg2_ems_adj_mat_list.append(cur_psg2_ems_adj_mat)

        cur_joint_ents = []
        cur_linked_entities = []

        cur_joint_ent = [0 for idx in range(psg1_max_ems_cnt + psg2_max_ems_cnt)]
        for idx in sample.Arg1EMs:
            cur_joint_ent[idx] = 1
        cur_joint_ents.append(cur_joint_ent)

        cur_joint_ent = [0 for idx in range(psg1_max_ems_cnt + psg2_max_ems_cnt)]
        for idx in sample.Arg2EMs:
            cur_joint_ent[psg1_max_ems_cnt + idx] = 1
        cur_joint_ents.append(cur_joint_ent)

        for i in range(0, len(sample.BothPsgEntEMs), 2):
            cur_joint_ent = [0 for idx in range(psg1_max_ems_cnt + psg2_max_ems_cnt)]
            for idx in sample.BothPsgEntEMs[i]:
                cur_joint_ent[idx] = 1
            for idx in sample.BothPsgEntEMs[i + 1]:
                cur_joint_ent[psg1_max_ems_cnt + idx] = 1
            cur_joint_ents.append(cur_joint_ent)
            cur_linked_entities.append(2 + i)

        cur_ems_paths = []
        for ems_e1_idx in sample.Arg1EMs:
            for ems_e2_idx in sample.Arg2EMs:
                for ems_link_idx in range(0, len(sample.BothPsgEntEMs), 2):
                    for ems_c1_idx in sample.BothPsgEntEMs[ems_link_idx]:
                        for ems_c2_idx in sample.BothPsgEntEMs[ems_link_idx + 1]:
                            cur_ems_paths.append((ems_e1_idx, ems_c1_idx, ems_c2_idx, ems_e2_idx))
        cur_ems_paths = cur_ems_paths[:min(batch_max_path_cnt, len(cur_ems_paths))]
        ems_path_mask_len = batch_max_path_cnt - len(cur_ems_paths)
        cur_ems_path_mask = [0 for i in range(len(cur_ems_paths))] + [1 for i in range(ems_path_mask_len)]
        ems_path_mask.append(cur_ems_path_mask)
        cur_ems_path_e1 = [sample_idx * psg1_max_ems_cnt + tup[0] for tup in cur_ems_paths] + \
                          [sample_idx * psg1_max_ems_cnt for i in range(ems_path_mask_len)]
        ems_path_e1.append(cur_ems_path_e1)
        cur_ems_path_c1 = [sample_idx * psg1_max_ems_cnt + tup[1] for tup in cur_ems_paths] + \
                          [sample_idx * psg1_max_ems_cnt for i in range(ems_path_mask_len)]
        ems_path_c1.append(cur_ems_path_c1)
        cur_ems_path_c2 = [sample_idx * psg2_max_ems_cnt + tup[2] for tup in cur_ems_paths] + \
                          [sample_idx * psg2_max_ems_cnt for i in range(ems_path_mask_len)]
        ems_path_c2.append(cur_ems_path_c2)
        cur_ems_path_e2 = [sample_idx * psg2_max_ems_cnt + tup[3] for tup in cur_ems_paths] + \
                          [sample_idx * psg2_max_ems_cnt for i in range(ems_path_mask_len)]
        ems_path_e2.append(cur_ems_path_e2)

        for i in range(len(sample.OnlyPsg1EntEMs)):
            cur_joint_ent = [0 for idx in range(psg1_max_ems_cnt + psg2_max_ems_cnt)]
            for idx in sample.OnlyPsg1EntEMs[i]:
                cur_joint_ent[idx] = 1
            cur_joint_ents.append(cur_joint_ent)

        for i in range(len(sample.OnlyPsg2EntEMs)):
            cur_joint_ent = [0 for idx in range(psg1_max_ems_cnt + psg2_max_ems_cnt)]
            for idx in sample.OnlyPsg2EntEMs[i]:
                cur_joint_ent[psg1_max_ems_cnt + idx] = 1
            cur_joint_ents.append(cur_joint_ent)

        cur_linked_ents = [sample_idx * joint_max_ent_cnt + idx for idx in cur_linked_entities] + \
                          [sample_idx * joint_max_ent_cnt for idx in range(joint_max_ent_cnt - len(cur_linked_entities))]
        linked_entities_list.append(cur_linked_ents)
        cur_linked_ents_mask = [0 for i in range(len(cur_linked_entities))] + \
                               [1 for i in range(joint_max_ent_cnt-len(cur_linked_entities))]
        linked_entities_mask.append(cur_linked_ents_mask)

        cur_joint_ents_sents_mask = []
        for lst in sample.EntsSents:
            cur_joint_ent_sents_mask = [1 for idx in range(psg1_max_sent + psg2_max_sent)]
            for tup in lst:
                if tup[0] == 1:
                    cur_joint_ent_sents_mask[tup[1]] = 0
                else:
                    cur_joint_ent_sents_mask[psg1_max_sent + tup[1]] = 0
            cur_joint_ents_sents_mask.append(cur_joint_ent_sents_mask)

        for i in range(joint_max_ent_cnt - len(cur_joint_ents)):
            cur_joint_ent_sents_mask = [0 for idx in range(psg1_max_sent + psg2_max_sent)]
            cur_joint_ents_sents_mask.append(cur_joint_ent_sents_mask)

        joint_ent_sent_mask_list.append(cur_joint_ents_sents_mask)

        cur_joint_ent_mask = [0 for i in range(len(cur_joint_ents))] + \
                             [1 for i in range(joint_max_ent_cnt - len(cur_joint_ents))]
        joint_ent_mask_list.append(cur_joint_ent_mask)

        cur_joint_ent_adj_mat = np.zeros((joint_max_ent_cnt, joint_max_ent_cnt), dtype=np.float32)
        cur_joint_ent_adj_mat[:len(cur_joint_ents), :len(cur_joint_ents)] = sample.MergedAdjMat

        joint_ent_adj_mat_list.append(cur_joint_ent_adj_mat)

        for i in range(joint_max_ent_cnt - len(cur_joint_ents)):
            cur_joint_ents.append([1 for idx in range(psg1_max_ems_cnt + psg2_max_ems_cnt)])

        joint_ent_list.append(cur_joint_ents)

        sample_idx += 1

        if is_training:
            rel_labels_list.append(rel_name_to_idx[sample.Relation])

    return {'psg1_words': np.array(psg1_words_list, dtype=np.float32),
            'psg1_mask': np.array(psg1_mask_list),
            'psg1_coref': np.array(psg1_coref_list),
            'psg2_words': np.array(psg2_words_list, dtype=np.float32),
            'psg2_mask': np.array(psg2_mask_list),
            'psg2_coref': np.array(psg2_coref_list),
            'arg1_start': np.array(arg1_s_list),
            'arg1_end': np.array(arg1_e_list),
            'arg2_start': np.array(arg2_s_list),
            'arg2_end': np.array(arg2_e_list),
            'arg1_mask': np.array(arg1_mask),
            'arg2_mask': np.array(arg2_mask),
            'path_e1_s': np.array(path_e1_s),
            'path_e1_e': np.array(path_e1_e),
            'path_c1_s': np.array(path_c1_s),
            'path_c1_e': np.array(path_c1_e),
            'path_c2_s': np.array(path_c2_s),
            'path_c2_e': np.array(path_c2_e),
            'path_e2_s': np.array(path_e2_s),
            'path_e2_e': np.array(path_e2_e),
            'path_mask': np.array(path_mask),
            'psg1_ems_start': np.array(psg1_ems_start_list),
            'psg1_ems_end': np.array(psg1_ems_end_list),
            'psg1_ems_mask': np.array(psg1_ems_mask_list),
            'psg1_ems_adj_mat': np.array(psg1_ems_adj_mat_list),
            'psg1_ems_words_mask': np.array(psg1_ems_words_mask_list),
            'psg2_ems_start': np.array(psg2_ems_start_list),
            'psg2_ems_end': np.array(psg2_ems_end_list),
            'psg2_ems_mask': np.array(psg2_ems_mask_list),
            'psg2_ems_adj_mat': np.array(psg2_ems_adj_mat_list),
            'psg2_ems_words_mask': np.array(psg2_ems_words_mask_list),
            'joint_ent': np.array(joint_ent_list),
            'joint_ent_mask': np.array(joint_ent_mask_list),
            'joint_ent_adj_mat': np.array(joint_ent_adj_mat_list),
            'linked_entities_map': np.array(linked_entities_list),
            'linked_entities_mask':np.array(linked_entities_mask),
            'ems_path_e1': np.array(ems_path_e1),
            'ems_path_c1': np.array(ems_path_c1),
            'ems_path_c2': np.array(ems_path_c2),
            'ems_path_e2': np.array(ems_path_e2),
            'ems_path_mask': np.array(ems_path_mask)}, \
            {'target': np.array(rel_labels_list, dtype=np.int32)}


# Calculate mean of the time steps

def mean_over_time(x, mask):
    x.masked_fill_(mask.unsqueeze(2), 0)
    x_sum = torch.sum(x, dim=1)
    x_len = torch.sum(mask.eq(0), 1)
    x_len = x_len.float()
    x_len[x_len == 0] = 1
    x_mean = torch.div(x_sum, x_len.unsqueeze(1))
    return x_mean


class cnn(nn.Module):
    def __init__(self, input_dim, num_filter, ngrams):
        super(cnn, self).__init__()
        self.NGrams = ngrams
        self.conv_layers = nn.ModuleList()
        for ngram in self.NGrams:
            self.conv_layers.append(nn.Conv1d(input_dim, num_filter, ngram))

    def forward(self, input, mask):
        mask = mask.unsqueeze(2)
        input.masked_fill_(mask, 0)
        input = input.permute(0, 2, 1)
        output = self.conv_layers[0](input)
        output = torch.tanh(torch.max(output, 2)[0])
        for i in range(1, len(self.NGrams)):
            cur_output = self.conv_layers[i](input)
            cur_output = torch.tanh(torch.max(cur_output, 2)[0])
            output = torch.cat((output, cur_output), -1)
        return output


# CNN Model definition

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.class_count = len(rel_name_to_idx)
        self.input_dim = word_embed_dim + other_embed_dim

        self.word_embeddings = nn.Embedding(len(word_vocab), word_embed_dim, padding_idx=0)
        if job_mode == 'train':
            self.word_embeddings.weight.data.copy_(torch.from_numpy(word_embed_matrix))
        self.coref_embeddings = nn.Embedding(max_coref_count, other_embed_dim, padding_idx=0)

        self.cnn = cnn(self.input_dim, cnn_filter_size, n_grams)
        self.classifier = nn.Linear(ngram_count * cnn_filter_size, self.class_count)
        self.hid_drop = nn.Dropout(hidden_dropout)
        self.visible_drop = nn.Dropout(visible_dropout)

    def forward(self, psg1_words_seq, psg1_words_mask, psg1_coref_seq,
                psg2_words_seq, psg2_words_mask, psg2_coref_seq,
                is_training=False):
        psg1_word_embeds = self.word_embeddings(psg1_words_seq)
        psg1_coref_embeds = self.coref_embeddings(psg1_coref_seq)
        psg2_word_embeds = self.word_embeddings(psg2_words_seq)
        psg2_coref_embeds = self.coref_embeddings(psg2_coref_seq)

        if apply_embed_dropout:
            psg1_word_embeds = self.visible_drop(psg1_word_embeds)
            psg1_coref_embeds = self.visible_drop(psg1_coref_embeds)
            psg2_word_embeds = self.visible_drop(psg2_word_embeds)
            psg2_coref_embeds = self.visible_drop(psg2_coref_embeds)

        psg1_input = torch.cat((psg1_word_embeds, psg1_coref_embeds), 2)
        psg2_input = torch.cat((psg2_word_embeds, psg2_coref_embeds), 2)
        psg_input = torch.cat((psg1_input, psg2_input), 1)
        psg_mask = torch.cat((psg1_words_mask, psg2_words_mask), -1)
        feature = self.cnn(psg_input, psg_mask)

        probs = self.classifier(self.hid_drop(feature))
        if is_training:
            probs = F.log_softmax(probs, dim=-1)
        else:
            probs = F.softmax(probs, dim=-1)
        return probs


# BiLSTM Model definition

class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.class_count = len(rel_name_to_idx)
        self.input_dim = word_embed_dim + other_embed_dim
        self.hidden_dim = 2 * self.input_dim

        self.word_embeddings = nn.Embedding(len(word_vocab), word_embed_dim, padding_idx=0)
        if job_mode == 'train':
            self.word_embeddings.weight.data.copy_(torch.from_numpy(word_embed_matrix))
        self.coref_embeddings = nn.Embedding(max_coref_count, other_embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(self.input_dim, int(self.hidden_dim/2), 1, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(4 * self.hidden_dim, self.class_count)
        self.hid_drop = nn.Dropout(hidden_dropout)
        self.visible_drop = nn.Dropout(visible_dropout)

    def forward(self, psg1_words_seq, psg1_words_mask, psg1_coref_seq, psg2_words_seq, psg2_words_mask,
                psg2_coref_seq, arg1_start, arg1_end, arg1_mask, arg2_start, arg2_end, arg2_mask,
                is_training=False):
        if is_training:
            self.lstm.flatten_parameters()
        psg1_word_embeds = self.word_embeddings(psg1_words_seq)
        psg2_word_embeds = self.word_embeddings(psg2_words_seq)
        psg1_coref_embeds = self.coref_embeddings(psg1_coref_seq)
        psg2_coref_embeds = self.coref_embeddings(psg2_coref_seq)

        if apply_embed_dropout:
            psg1_word_embeds = self.visible_drop(psg1_word_embeds)
            psg2_word_embeds = self.visible_drop(psg2_word_embeds)
            psg1_coref_embeds = self.visible_drop(psg1_coref_embeds)
            psg2_coref_embeds = self.visible_drop(psg2_coref_embeds)

        psg1_lstm_input = torch.cat((psg1_word_embeds, psg1_coref_embeds), 2)
        psg1_lstm_input.masked_fill_(psg1_words_mask.unsqueeze(2), 0)

        psg2_lstm_input = torch.cat((psg2_word_embeds, psg2_coref_embeds), 2)
        psg2_lstm_input.masked_fill_(psg2_words_mask.unsqueeze(2), 0)

        psg_input = torch.cat((psg1_lstm_input, psg2_lstm_input), 1)
        psg_output, _ = self.lstm(psg_input)
        psg_output = psg_output.contiguous()
        psg_mask =  torch.cat((psg1_words_mask, psg2_words_mask), 1)
        psg_output.masked_fill_(psg_mask.unsqueeze(2), 0)

        psg1_output = psg_output[:, 0:psg1_word_embeds.size()[1], :]
        psg2_output = psg_output[:, psg1_word_embeds.size()[1]:, :]
        psg1_output = psg1_output.contiguous()
        psg2_output = psg2_output.contiguous()
        psg1 = psg1_output.view(-1, psg1_output.size()[-1])
        psg2 = psg2_output.view(-1, psg2_output.size()[-1])

        arg1_s_vec = torch.index_select(psg1, 0, arg1_start.view(-1)).view(arg1_start.size()[0],
                                                                           arg1_start.size()[1], -1)
        arg1_e_vec = torch.index_select(psg1, 0, arg1_end.view(-1)).view(arg1_end.size()[0],
                                                                         arg1_end.size()[1], -1)
        arg1 = torch.cat((arg1_s_vec, arg1_e_vec), -1)
        arg1.masked_fill_(arg1_mask.unsqueeze(2), 0)
        arg1 = torch.sum(arg1, 1) / torch.sum(arg1_mask.eq(0).float(), -1, keepdim=True)

        arg2_s_vec = torch.index_select(psg2, 0, arg2_start.view(-1)).view(arg2_start.size()[0],
                                                                           arg2_start.size()[1], -1)
        arg2_e_vec = torch.index_select(psg2, 0, arg2_end.view(-1)).view(arg2_end.size()[0],
                                                                         arg2_end.size()[1], -1)
        arg2 = torch.cat((arg2_s_vec, arg2_e_vec), -1)
        arg2.masked_fill_(arg2_mask.unsqueeze(2), 0)
        arg2 = torch.sum(arg2, 1) / torch.sum(arg2_mask.eq(0).float(), -1, keepdim=True)

        probs = self.classifier(self.hid_drop(torch.cat((arg1, arg2), -1)))
        if is_training:
            probs = F.log_softmax(probs, dim=-1)
        else:
            probs = F.softmax(probs, dim=-1)
        return probs


# BiLSTM_CNN Model definition

class BiLSTM_CNN(nn.Module):
    def __init__(self):
        super(BiLSTM_CNN, self).__init__()
        self.class_count = len(rel_name_to_idx)
        self.input_dim = word_embed_dim + other_embed_dim
        self.hidden_dim = 2 * self.input_dim

        self.word_embeddings = nn.Embedding(len(word_vocab), word_embed_dim, padding_idx=0)
        if job_mode == 'train':
            self.word_embeddings.weight.data.copy_(torch.from_numpy(word_embed_matrix))
        self.coref_embeddings = nn.Embedding(max_coref_count, other_embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(self.input_dim, int(self.hidden_dim/2), 1, batch_first=True, bidirectional=True)
        self.cnn = cnn(self.hidden_dim, cnn_filter_size, n_grams)
        self.classifier = nn.Linear(ngram_count * cnn_filter_size, self.class_count)
        self.hid_drop = nn.Dropout(hidden_dropout)
        self.visible_drop = nn.Dropout(visible_dropout)

    def forward(self, psg1_words_seq, psg1_words_mask, psg1_coref_seq,
                psg2_words_seq, psg2_words_mask, psg2_coref_seq,
                is_training=False):
        if is_training:
            self.lstm.flatten_parameters()
        psg1_word_embeds = self.word_embeddings(psg1_words_seq)
        psg1_coref_embeds = self.coref_embeddings(psg1_coref_seq)
        psg2_word_embeds = self.word_embeddings(psg2_words_seq)
        psg2_coref_embeds = self.coref_embeddings(psg2_coref_seq)

        if apply_embed_dropout:
            psg1_word_embeds = self.visible_drop(psg1_word_embeds)
            psg1_coref_embeds = self.visible_drop(psg1_coref_embeds)
            psg2_word_embeds = self.visible_drop(psg2_word_embeds)
            psg2_coref_embeds = self.visible_drop(psg2_coref_embeds)

        psg1_lstm_input = torch.cat((psg1_word_embeds, psg1_coref_embeds), 2)
        psg1_lstm_input.masked_fill_(psg1_words_mask.unsqueeze(2), 0)

        psg2_lstm_input = torch.cat((psg2_word_embeds, psg2_coref_embeds), 2)
        psg2_lstm_input.masked_fill_(psg2_words_mask.unsqueeze(2), 0)

        psg_input = torch.cat((psg1_lstm_input, psg2_lstm_input), 1)
        psg_output, _ = self.lstm(psg_input)
        psg_output = psg_output.contiguous()
        psg_mask = torch.cat((psg1_words_mask, psg2_words_mask), 1)
        psg_output.masked_fill_(psg_mask.unsqueeze(2), 0)

        feature = self.cnn(psg_output, psg_mask)
        probs = self.classifier(self.hid_drop(feature))
        if is_training:
            probs = F.log_softmax(probs, dim=-1)
        else:
            probs = F.softmax(probs, dim=-1)
        return probs


# LinkPath Model definition

class LinkPath(nn.Module):
    def __init__(self):
        super(LinkPath, self).__init__()
        self.class_count = len(rel_name_to_idx)
        self.input_dim = word_embed_dim + other_embed_dim
        self.hidden_dim = 2 * self.input_dim

        self.word_embeddings = nn.Embedding(len(word_vocab), word_embed_dim, padding_idx=0)
        if job_mode == 'train':
            self.word_embeddings.weight.data.copy_(torch.from_numpy(word_embed_matrix))
        self.coref_embeddings = nn.Embedding(max_coref_count, other_embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(self.input_dim, int(self.hidden_dim/2), 1, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(8 * self.hidden_dim, self.class_count)
        self.hid_drop = nn.Dropout(hidden_dropout)
        self.visible_drop = nn.Dropout(visible_dropout)

    def forward(self, psg1_words_seq, psg1_words_mask, psg1_coref_seq, psg2_words_seq, psg2_words_mask, psg2_coref_seq,
                path_e1_s, path_e1_e, path_c1_s, path_c1_e, path_c2_s, path_c2_e, path_e2_s, path_e2_e, path_mask,
                is_training=False):
        if is_training:
            self.lstm.flatten_parameters()
        psg1_word_embeds = self.word_embeddings(psg1_words_seq)
        psg2_word_embeds = self.word_embeddings(psg2_words_seq)

        psg1_coref_embeds = self.coref_embeddings(psg1_coref_seq)
        psg2_coref_embeds = self.coref_embeddings(psg2_coref_seq)

        if apply_embed_dropout:
            psg1_word_embeds = self.visible_drop(psg1_word_embeds)
            psg2_word_embeds = self.visible_drop(psg2_word_embeds)
            psg1_coref_embeds = self.visible_drop(psg1_coref_embeds)
            psg2_coref_embeds = self.visible_drop(psg2_coref_embeds)

        psg1_input = torch.cat((psg1_word_embeds, psg1_coref_embeds), 2)
        psg1_input.masked_fill_(psg1_words_mask.unsqueeze(2), 0)

        psg2_input = torch.cat((psg2_word_embeds, psg2_coref_embeds), 2)
        psg2_input.masked_fill_(psg2_words_mask.unsqueeze(2), 0)

        psg_input = torch.cat((psg1_input, psg2_input), 1)
        psg_output, _ = self.lstm(psg_input)
        psg_output = psg_output.contiguous()
        psg_mask = torch.cat((psg1_words_mask, psg2_words_mask), 1)
        psg_output.masked_fill_(psg_mask.unsqueeze(2), 0)

        psg1_output = psg_output[:, 0:psg1_word_embeds.size()[1], :]
        psg2_output = psg_output[:, psg1_word_embeds.size()[1]:, :]
        psg1_output = psg1_output.contiguous()
        psg2_output = psg2_output.contiguous()

        psg1 = psg1_output.view(-1, psg1_output.size()[-1])
        psg2 = psg2_output.view(-1, psg2_output.size()[-1])

        path_e1_s_vec = torch.index_select(psg1, 0, path_e1_s.view(-1)).view(path_e1_s.size()[0],
                                                                             path_e1_s.size()[1], -1)
        path_e1_e_vec = torch.index_select(psg1, 0, path_e1_e.view(-1)).view(path_e1_s.size()[0],
                                                                             path_e1_s.size()[1], -1)
        path_c1_s_vec = torch.index_select(psg1, 0, path_c1_s.view(-1)).view(path_e1_s.size()[0],
                                                                             path_e1_s.size()[1], -1)
        path_c1_e_vec = torch.index_select(psg1, 0, path_c1_e.view(-1)).view(path_e1_s.size()[0],
                                                                             path_e1_s.size()[1], -1)
        path_e1 = torch.cat((path_e1_s_vec, path_e1_e_vec), -1)
        path_c1 = torch.cat((path_c1_s_vec, path_c1_e_vec), -1)
        p1_rel = torch.cat((path_e1, path_c1), -1)

        path_c2_s_vec = torch.index_select(psg2, 0, path_c2_s.view(-1)).view(path_e1_s.size()[0],
                                                                             path_e1_s.size()[1], -1)
        path_c2_e_vec = torch.index_select(psg2, 0, path_c2_e.view(-1)).view(path_e1_s.size()[0],
                                                                             path_e1_s.size()[1], -1)
        path_e2_s_vec = torch.index_select(psg2, 0, path_e2_s.view(-1)).view(path_e1_s.size()[0],
                                                                             path_e1_s.size()[1], -1)
        path_e2_e_vec = torch.index_select(psg2, 0, path_e2_e.view(-1)).view(path_e1_s.size()[0],
                                                                             path_e1_s.size()[1], -1)
        path_c2 = torch.cat((path_c2_s_vec, path_c2_e_vec), -1)
        path_e2 = torch.cat((path_e2_s_vec, path_e2_e_vec), -1)
        p2_rel = torch.cat((path_e2, path_c2), -1)

        paths = torch.cat((p1_rel, p2_rel), -1)
        paths.masked_fill_(path_mask.unsqueeze(2), 0)
        path = torch.sum(paths, 1) / torch.sum(path_mask.eq(0).float(), -1, keepdim=True)
        probs = self.classifier(self.hid_drop(path))
        # probs = torch.max(probs, 1)[0]

        if is_training:
            probs = F.log_softmax(probs, dim=-1)
        else:
            probs = F.softmax(probs, dim=-1)
        return probs


# GCN Layer

class GCN(nn.Module):
    def __init__(self, num_layers, in_dim, out_dim):
        super(GCN, self).__init__()
        self.gcn_num_layers = num_layers
        self.gcn_layers = nn.ModuleList()
        for i in range(self.gcn_num_layers):
            self.gcn_layers.append(nn.Linear(in_dim, out_dim))
        self.hid_drop = nn.Dropout(hidden_dropout)

    def forward(self, gcn_input, adj):
        for i in range(0, self.gcn_num_layers):
            gcn_input = self.gcn_layers[i](gcn_input)
            gcn_input = torch.bmm(adj, gcn_input)
            gcn_input = F.relu(gcn_input)
            if i < self.gcn_num_layers - 1:
                gcn_input = self.hid_drop(gcn_input)
        return gcn_input


# Attention layer for entity context vector

class EM_Word_Attention(nn.Module):
    def __init__(self, ems_dim, word_dim):
        super(EM_Word_Attention, self).__init__()
        self.w = nn.Linear(ems_dim, word_dim)

    def forward(self, ems, words, ems_words_mask):
        att_scores = torch.bmm(torch.tanh(self.w(ems)), words.transpose(1, 2))
        att_scores = att_scores.masked_fill(ems_words_mask, -float('inf'))
        att_scores = F.softmax(att_scores, -1)
        ems_ctx = torch.bmm(att_scores, words)
        return ems_ctx


def EM_Word_Average(words, ems_words_mask):
    mask_copy = ems_words_mask.clone()
    mask_copy[ems_words_mask==0] = 1
    mask_copy[ems_words_mask==1] = 0
    mask_copy = mask_copy.float()
    ems_ctx = torch.bmm(mask_copy, words)/torch.sum(mask_copy, 2, keepdim=True).clamp(min=1)
    return ems_ctx


# HEGCN Model definition

class EntityGCN(nn.Module):
    def __init__(self):
        super(EntityGCN, self).__init__()
        self.class_count = len(rel_name_to_idx)
        self.input_dim = word_embed_dim + other_embed_dim
        self.hidden_dim = 2 * self.input_dim

        self.word_embeddings = nn.Embedding(len(word_vocab), word_embed_dim, padding_idx=0)
        if job_mode == 'train':
            self.word_embeddings.weight.data.copy_(torch.from_numpy(word_embed_matrix))
        self.coref_embeddings = nn.Embedding(max_coref_count, other_embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(self.input_dim, int(self.hidden_dim/2), 1, batch_first=True, bidirectional=True)
        if ems_gcn_layers_cnt > 0:
            self.ems_gcn = GCN(ems_gcn_layers_cnt, 3 * self.hidden_dim, 3 * self.hidden_dim)
        if att_type == 1:
            self.word_att = EM_Word_Attention(2 * self.hidden_dim, self.hidden_dim)
        if ent_gcn_layers_cnt > 0:
            self.ent_gcn = GCN(ent_gcn_layers_cnt, 3 * self.hidden_dim, 3 * self.hidden_dim)
        self.classifier = nn.Linear(6 * self.hidden_dim, self.class_count)
        self.hid_drop = nn.Dropout(hidden_dropout)
        self.visible_drop = nn.Dropout(visible_dropout)

    def forward(self, psg1_words_seq, psg1_words_mask, psg1_coref_seq, psg2_words_seq, psg2_words_mask, psg2_coref_seq,
                psg1_ems_start, psg1_ems_end, psg1_ems_mask, psg1_ems_adj_mat, psg2_ems_start, psg2_ems_end,
                psg2_ems_mask, psg2_ems_adj_mat, joint_ent_ems, joint_ent_mask, joint_ent_adj_mat,
                psg1_ems_words_mask, psg2_ems_words_mask, is_training=False):
        if is_training:
            self.lstm.flatten_parameters()
        psg1_word_embeds = self.word_embeddings(psg1_words_seq)
        psg1_coref_embeds = self.coref_embeddings(psg1_coref_seq)
        psg2_word_embeds = self.word_embeddings(psg2_words_seq)
        psg2_coref_embeds = self.coref_embeddings(psg2_coref_seq)

        if apply_embed_dropout:
            psg1_word_embeds = self.visible_drop(psg1_word_embeds)
            psg1_coref_embeds = self.visible_drop(psg1_coref_embeds)
            psg2_word_embeds = self.visible_drop(psg2_word_embeds)
            psg2_coref_embeds = self.visible_drop(psg2_coref_embeds)

        psg1_input = torch.cat((psg1_word_embeds, psg1_coref_embeds), 2)
        psg1_input.masked_fill_(psg1_words_mask.unsqueeze(2), 0)

        psg2_input = torch.cat((psg2_word_embeds, psg2_coref_embeds), 2)
        psg2_input.masked_fill_(psg2_words_mask.unsqueeze(2), 0)

        psg_input = torch.cat((psg1_input, psg2_input), 1)
        psg_output, _ = self.lstm(psg_input)
        psg_output = psg_output.contiguous()
        psg_mask = torch.cat((psg1_words_mask, psg2_words_mask), 1)
        psg_output.masked_fill_(psg_mask.unsqueeze(2), 0)

        psg1_output = psg_output[:, 0:psg1_word_embeds.size()[1], :]
        psg2_output = psg_output[:, psg1_word_embeds.size()[1]:, :]
        psg1_output = psg1_output.contiguous()
        psg2_output = psg2_output.contiguous()

        psg1 = psg1_output.view(-1, psg1_output.size()[-1])
        psg2 = psg2_output.view(-1, psg2_output.size()[-1])

        psg1_ems_s_vec = torch.index_select(psg1, 0, psg1_ems_start.view(-1)).view(psg1_ems_start.size()[0],
                                                                                   psg1_ems_start.size()[1], -1)
        psg1_ems_e_vec = torch.index_select(psg1, 0, psg1_ems_end.view(-1)).view(psg1_ems_end.size()[0],
                                                                                 psg1_ems_end.size()[1], -1)
        psg1_ems = torch.cat((psg1_ems_s_vec, psg1_ems_e_vec), -1)
        psg1_ems.masked_fill_(psg1_ems_mask.unsqueeze(2), 0)
        if att_type == 1:
            psg1_ems_ctx = self.word_att(psg1_ems, psg1_output, psg1_ems_words_mask)
        else:
            psg1_ems_ctx = EM_Word_Average(psg1_output, psg1_ems_words_mask)
        psg1_ems = torch.cat((psg1_ems, psg1_ems_ctx), -1)
        psg1_ems.masked_fill_(psg1_ems_mask.unsqueeze(2), 0)
        if ems_gcn_layers_cnt > 0:
            psg1_ems = self.ems_gcn(psg1_ems, psg1_ems_adj_mat)

        psg2_ems_s_vec = torch.index_select(psg2, 0, psg2_ems_start.view(-1)).view(psg2_ems_start.size()[0],
                                                                                   psg2_ems_start.size()[1], -1)
        psg2_ems_e_vec = torch.index_select(psg2, 0, psg2_ems_end.view(-1)).view(psg2_ems_end.size()[0],
                                                                                 psg2_ems_end.size()[1], -1)
        psg2_ems = torch.cat((psg2_ems_s_vec, psg2_ems_e_vec), -1)
        psg2_ems.masked_fill_(psg2_ems_mask.unsqueeze(2), 0)
        if att_type == 1:
            psg2_ems_ctx = self.word_att(psg2_ems, psg2_output, psg2_ems_words_mask)
        else:
            psg2_ems_ctx = EM_Word_Average(psg2_output, psg2_ems_words_mask)
        psg2_ems = torch.cat((psg2_ems, psg2_ems_ctx), -1)
        psg2_ems.masked_fill_(psg2_ems_mask.unsqueeze(2), 0)
        if ems_gcn_layers_cnt > 0:
            psg2_ems = self.ems_gcn(psg2_ems, psg2_ems_adj_mat)

        joint_ems = torch.cat((psg1_ems, psg2_ems), 1)
        joint_ent = torch.bmm(joint_ent_ems, joint_ems) / torch.sum(joint_ent_ems, -1, keepdim=True)
        joint_ent.masked_fill_(joint_ent_mask.unsqueeze(2), 0)
        if ent_gcn_layers_cnt > 0:
            joint_ent = self.ent_gcn(joint_ent, joint_ent_adj_mat)

        ent1 = joint_ent[:, 0, :].squeeze()
        ent2 = joint_ent[:, 1, :].squeeze()

        probs = self.classifier(self.hid_drop(torch.cat((ent1, ent2), -1)))
        if is_training:
            probs = F.log_softmax(probs, dim=-1)
        else:
            probs = F.softmax(probs, dim=-1)
        return probs


def get_model(model_id):
    if model_id == 1:
        return CNN()
    if model_id == 2:
        return BiLSTM()
    if model_id == 3:
        return BiLSTM_CNN()
    if model_id == 4:
        return LinkPath()
    if model_id == 5:
        return EntityGCN()


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Function for prediction from a trained model

def predict(samples, model, model_id):
    batch_count = math.ceil(len(samples) / pred_batch_size)
    move_last_batch = False
    if len(samples) - pred_batch_size * (batch_count - 1) == 1:
        move_last_batch = True
        batch_count -= 1
    preds = list()
    model.eval()
    set_random_seeds(random_seed)

    for batch_idx in tqdm(range(0, batch_count)):
        batch_start = batch_idx * pred_batch_size
        batch_end = min(len(samples), batch_start + pred_batch_size)
        if batch_idx == batch_count - 1 and move_last_batch:
            batch_end = len(samples)

        cur_batch = samples[batch_start:batch_end]
        cur_input, _ = get_batch_data(cur_batch, True)

        psg1_words_seq = autograd.Variable(torch.from_numpy(cur_input['psg1_words'].astype('long')).cuda())
        psg1_words_mask = autograd.Variable(torch.from_numpy(cur_input['psg1_mask'].astype('bool')).cuda())
        psg1_coref_seq = autograd.Variable(torch.from_numpy(cur_input['psg1_coref'].astype('long')).cuda())
        psg2_words_seq = autograd.Variable(torch.from_numpy(cur_input['psg2_words'].astype('long')).cuda())
        psg2_words_mask = autograd.Variable(torch.from_numpy(cur_input['psg2_mask'].astype('bool')).cuda())
        psg2_coref_seq = autograd.Variable(torch.from_numpy(cur_input['psg2_coref'].astype('long')).cuda())
        path_mask = autograd.Variable(torch.from_numpy(cur_input['path_mask'].astype('bool')).cuda())
        path_e1_s = autograd.Variable(torch.from_numpy(cur_input['path_e1_s'].astype('long')).cuda())
        path_e1_e = autograd.Variable(torch.from_numpy(cur_input['path_e1_e'].astype('long')).cuda())
        path_c1_s = autograd.Variable(torch.from_numpy(cur_input['path_c1_s'].astype('long')).cuda())
        path_c1_e = autograd.Variable(torch.from_numpy(cur_input['path_c1_e'].astype('long')).cuda())
        path_c2_s = autograd.Variable(torch.from_numpy(cur_input['path_c2_s'].astype('long')).cuda())
        path_c2_e = autograd.Variable(torch.from_numpy(cur_input['path_c2_e'].astype('long')).cuda())
        path_e2_s = autograd.Variable(torch.from_numpy(cur_input['path_e2_s'].astype('long')).cuda())
        path_e2_e = autograd.Variable(torch.from_numpy(cur_input['path_e2_e'].astype('long')).cuda())
        arg1_s = autograd.Variable(torch.from_numpy(cur_input['arg1_start'].astype('long')).cuda())
        arg1_e = autograd.Variable(torch.from_numpy(cur_input['arg1_end'].astype('long')).cuda())
        arg2_s = autograd.Variable(torch.from_numpy(cur_input['arg2_start'].astype('long')).cuda())
        arg2_e = autograd.Variable(torch.from_numpy(cur_input['arg2_end'].astype('long')).cuda())
        arg1_mask = autograd.Variable(torch.from_numpy(cur_input['arg1_mask'].astype('bool')).cuda())
        arg2_mask = autograd.Variable(torch.from_numpy(cur_input['arg2_mask'].astype('bool')).cuda())
        psg1_ems_start = autograd.Variable(torch.from_numpy(cur_input['psg1_ems_start'].astype('long')).cuda())
        psg1_ems_end = autograd.Variable(torch.from_numpy(cur_input['psg1_ems_end'].astype('long')).cuda())
        psg1_ems_mask = autograd.Variable(torch.from_numpy(cur_input['psg1_ems_mask'].astype('bool')).cuda())
        psg1_ems_adj_mat = autograd.Variable(torch.from_numpy(cur_input['psg1_ems_adj_mat'].astype('float32')).cuda())
        psg1_ems_words_mask = autograd.Variable(torch.from_numpy(cur_input['psg1_ems_words_mask'].astype('bool')).cuda())
        psg2_ems_start = autograd.Variable(torch.from_numpy(cur_input['psg2_ems_start'].astype('long')).cuda())
        psg2_ems_end = autograd.Variable(torch.from_numpy(cur_input['psg2_ems_end'].astype('long')).cuda())
        psg2_ems_mask = autograd.Variable(torch.from_numpy(cur_input['psg2_ems_mask'].astype('bool')).cuda())
        psg2_ems_adj_mat = autograd.Variable(torch.from_numpy(cur_input['psg2_ems_adj_mat'].astype('float32')).cuda())
        psg2_ems_words_mask = autograd.Variable(torch.from_numpy(cur_input['psg2_ems_words_mask'].astype('bool')).cuda())
        joint_ent = autograd.Variable(torch.from_numpy(cur_input['joint_ent'].astype('float32')).cuda())
        joint_ent_mask = autograd.Variable(torch.from_numpy(cur_input['joint_ent_mask'].astype('bool')).cuda())
        joint_ent_adj_mat = autograd.Variable(torch.from_numpy(cur_input['joint_ent_adj_mat'].astype('float32')).cuda())

        with torch.no_grad():
            if model_id in [1, 3]:
                outputs = model(psg1_words_seq, psg1_words_mask, psg1_coref_seq,
                                psg2_words_seq, psg2_words_mask, psg2_coref_seq,
                                False)
            elif model_id in [2]:
                outputs = model(psg1_words_seq, psg1_words_mask, psg1_coref_seq,
                                psg2_words_seq, psg2_words_mask, psg2_coref_seq,
                                arg1_s, arg1_e, arg1_mask, arg2_s, arg2_e, arg2_mask, False)
            elif model_id in [4]:
                outputs = model(psg1_words_seq, psg1_words_mask, psg1_coref_seq,
                                psg2_words_seq, psg2_words_mask, psg2_coref_seq,
                                path_e1_s, path_e1_e, path_c1_s, path_c1_e, path_c2_s,
                                path_c2_e, path_e2_s, path_e2_e, path_mask, False)
            elif model_id in [5]:
                outputs = model(psg1_words_seq, psg1_words_mask, psg1_coref_seq,
                                psg2_words_seq, psg2_words_mask, psg2_coref_seq,
                                psg1_ems_start, psg1_ems_end, psg1_ems_mask, psg1_ems_adj_mat,
                                psg2_ems_start, psg2_ems_end, psg2_ems_mask, psg2_ems_adj_mat,
                                joint_ent, joint_ent_mask, joint_ent_adj_mat,
                                psg1_ems_words_mask, psg2_ems_words_mask, False)

        preds += list(outputs.data.cpu().numpy())
    model.zero_grad()
    return preds


# function for training a model

def torch_train(model_id, train_samples, dev_samples, test_samples, model_file):
    random.shuffle(train_samples)
    train_size = len(train_samples)
    batch_count = int(math.ceil(train_size/batch_size))
    move_last_batch = False
    if train_size - batch_size * (batch_count - 1) == 1:
        move_last_batch = True
        batch_count -= 1
    custom_print(batch_count)
    model = get_model(model_id)
    custom_print(model)
    if job_mode == 'train':
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        custom_print('Parameters size:', pytorch_total_params)

    if torch.cuda.is_available():
        model.cuda()
        model = torch.nn.DataParallel(model)

    criterion = nn.NLLLoss()
    optimizer = optim.Adagrad(model.parameters())
    custom_print(optimizer)

    best_dev_acc = -1.0
    best_epoch_idx = -1
    best_th = -1.0
    for epoch_idx in range(0, num_epoch):
        model.train()
        model.zero_grad()
        custom_print('Epoch:', epoch_idx + 1)
        custom_print('Model id:', model_id)
        cur_seed = random_seed + epoch_idx + 1
        set_random_seeds(cur_seed)
        # cur_shuffled_train_data = train_samples
        random.shuffle(train_samples)
        start_time = datetime.datetime.now()
        train_loss_val = 0.0
        for batch_idx in tqdm(range(0, batch_count)):
            # print(batch_idx + 1)
            batch_start = batch_idx * batch_size
            batch_end = min(train_size, batch_start + batch_size)
            if batch_idx == batch_count - 1 and move_last_batch:
                batch_end = train_size

            cur_batch = train_samples[batch_start:batch_end]
            cur_input, cur_target = get_batch_data(cur_batch, True)

            psg1_words_seq = autograd.Variable(torch.from_numpy(cur_input['psg1_words'].astype('long')).cuda())
            psg1_words_mask = autograd.Variable(torch.from_numpy(cur_input['psg1_mask'].astype('bool')).cuda())
            psg1_coref_seq = autograd.Variable(torch.from_numpy(cur_input['psg1_coref'].astype('long')).cuda())
            psg2_words_seq = autograd.Variable(torch.from_numpy(cur_input['psg2_words'].astype('long')).cuda())
            psg2_words_mask = autograd.Variable(torch.from_numpy(cur_input['psg2_mask'].astype('bool')).cuda())
            psg2_coref_seq = autograd.Variable(torch.from_numpy(cur_input['psg2_coref'].astype('long')).cuda())
            arg1_s = autograd.Variable(torch.from_numpy(cur_input['arg1_start'].astype('long')).cuda())
            arg1_e = autograd.Variable(torch.from_numpy(cur_input['arg1_end'].astype('long')).cuda())
            arg2_s = autograd.Variable(torch.from_numpy(cur_input['arg2_start'].astype('long')).cuda())
            arg2_e = autograd.Variable(torch.from_numpy(cur_input['arg2_end'].astype('long')).cuda())
            arg1_mask = autograd.Variable(torch.from_numpy(cur_input['arg1_mask'].astype('bool')).cuda())
            arg2_mask = autograd.Variable(torch.from_numpy(cur_input['arg2_mask'].astype('bool')).cuda())
            path_e1_s = autograd.Variable(torch.from_numpy(cur_input['path_e1_s'].astype('long')).cuda())
            path_e1_e = autograd.Variable(torch.from_numpy(cur_input['path_e1_e'].astype('long')).cuda())
            path_c1_s = autograd.Variable(torch.from_numpy(cur_input['path_c1_s'].astype('long')).cuda())
            path_c1_e = autograd.Variable(torch.from_numpy(cur_input['path_c1_e'].astype('long')).cuda())
            path_c2_s = autograd.Variable(torch.from_numpy(cur_input['path_c2_s'].astype('long')).cuda())
            path_c2_e = autograd.Variable(torch.from_numpy(cur_input['path_c2_e'].astype('long')).cuda())
            path_e2_s = autograd.Variable(torch.from_numpy(cur_input['path_e2_s'].astype('long')).cuda())
            path_e2_e = autograd.Variable(torch.from_numpy(cur_input['path_e2_e'].astype('long')).cuda())
            path_mask = autograd.Variable(torch.from_numpy(cur_input['path_mask'].astype('bool')).cuda())
            psg1_ems_start = autograd.Variable(torch.from_numpy(cur_input['psg1_ems_start'].astype('long')).cuda())
            psg1_ems_end = autograd.Variable(torch.from_numpy(cur_input['psg1_ems_end'].astype('long')).cuda())
            psg1_ems_mask = autograd.Variable(torch.from_numpy(cur_input['psg1_ems_mask'].astype('bool')).cuda())
            psg1_ems_adj_mat = autograd.Variable(torch.from_numpy(cur_input['psg1_ems_adj_mat'].astype('float32')).cuda())
            psg1_ems_words_mask = autograd.Variable(torch.from_numpy(cur_input['psg1_ems_words_mask'].astype('bool')).cuda())
            psg2_ems_start = autograd.Variable(torch.from_numpy(cur_input['psg2_ems_start'].astype('long')).cuda())
            psg2_ems_end = autograd.Variable(torch.from_numpy(cur_input['psg2_ems_end'].astype('long')).cuda())
            psg2_ems_mask = autograd.Variable(torch.from_numpy(cur_input['psg2_ems_mask'].astype('bool')).cuda())
            psg2_ems_adj_mat = autograd.Variable(torch.from_numpy(cur_input['psg2_ems_adj_mat'].astype('float32')).cuda())
            psg2_ems_words_mask = autograd.Variable(torch.from_numpy(cur_input['psg2_ems_words_mask'].astype('bool')).cuda())
            joint_ent = autograd.Variable(torch.from_numpy(cur_input['joint_ent'].astype('float32')).cuda())
            joint_ent_mask = autograd.Variable(torch.from_numpy(cur_input['joint_ent_mask'].astype('bool')).cuda())
            joint_ent_adj_mat = autograd.Variable(torch.from_numpy(cur_input['joint_ent_adj_mat'].astype('float32')).cuda())

            target = autograd.Variable(torch.from_numpy(cur_target['target'].astype('long')).cuda())

            with torch.autograd.set_detect_anomaly(True):
                if model_id in [1, 3]:
                    outputs = model(psg1_words_seq, psg1_words_mask, psg1_coref_seq,
                                    psg2_words_seq, psg2_words_mask, psg2_coref_seq,
                                    True)
                elif model_id in [2]:
                    outputs = model(psg1_words_seq, psg1_words_mask, psg1_coref_seq,
                                    psg2_words_seq, psg2_words_mask, psg2_coref_seq,
                                    arg1_s, arg1_e, arg1_mask, arg2_s, arg2_e, arg2_mask, True)
                elif model_id in [4]:

                    outputs = model(psg1_words_seq, psg1_words_mask, psg1_coref_seq,
                                    psg2_words_seq, psg2_words_mask, psg2_coref_seq,
                                    path_e1_s, path_e1_e, path_c1_s, path_c1_e, path_c2_s,
                                    path_c2_e, path_e2_s, path_e2_e, path_mask, True)
                elif model_id in [5]:
                    outputs = model(psg1_words_seq, psg1_words_mask, psg1_coref_seq,
                                    psg2_words_seq, psg2_words_mask, psg2_coref_seq,
                                    psg1_ems_start, psg1_ems_end, psg1_ems_mask, psg1_ems_adj_mat,
                                    psg2_ems_start, psg2_ems_end, psg2_ems_mask, psg2_ems_adj_mat,
                                    joint_ent, joint_ent_mask, joint_ent_adj_mat,
                                    psg1_ems_words_mask, psg2_ems_words_mask, True)

            loss = criterion(outputs, target)
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if ((batch_idx + 1) % update_freq == 0) or batch_idx == batch_count - 1:
                optimizer.step()
                model.zero_grad()
            train_loss_val += loss.item()
            # print(train_loss_val)

        train_loss_val /= batch_count
        end_time = datetime.datetime.now()
        custom_print('Training Loss:', train_loss_val)
        custom_print('Time:', end_time - start_time)

        custom_print('\nDev Results\n')
        set_random_seeds(random_seed)
        dev_preds = predict(dev_samples, model, model_id)
        dev_th, dev_acc = get_f1_and_threshold(dev_samples, dev_preds)
        custom_print('RE Threshold:', dev_th)
        custom_print('Dev RE F1:', dev_acc)

        if dev_acc > best_dev_acc:
            best_epoch_idx = epoch_idx + 1
            custom_print('model saved......')
            best_dev_acc = dev_acc
            best_th = dev_th
            torch.save(model.state_dict(), model_file)

        custom_print('\nTest Results\n')
        set_random_seeds(random_seed)
        test_preds = predict(test_samples, model, model_id)
        re_f1 = get_F1(test_samples, test_preds, dev_th)
        custom_print('Test RE F1:', re_f1)

        custom_print('\n\n')
        if epoch_idx + 1 - best_epoch_idx >= early_stop_cnt:
            break
    custom_print('*******')
    custom_print('Best Epoch:', best_epoch_idx)
    custom_print('Best RE Threshold:', best_th)


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    random_seed = 1023
    set_random_seeds(random_seed)
    src_data_folder = sys.argv[1]
    # glove 6B.300d path
    embedding_file = sys.argv[2]
    trg_data_folder = sys.argv[3]
    if not os.path.exists(trg_data_folder):
        os.mkdir(trg_data_folder)
    model_name = int(sys.argv[4])
    job_mode = sys.argv[5]
    logger = None
    if job_mode == 'train':
        logger = open(os.path.join(trg_data_folder, 'training.log'), 'w')
    if job_mode == 'test':
        logger = open(os.path.join(trg_data_folder, 'test.log'), 'w')
    batch_size = 32
    pred_batch_size = batch_size
    num_epoch = 30

    neg_size = 1  # int(sys.argv[11])
    update_freq = 1
    use_coref = True
    ems_gcn_layers_cnt = 1
    ent_gcn_layers_cnt = 1
    att_type = 1
    emg_type1 = True
    emg_type2 = True
    emg_type3 = True
    eg_type1 = True
    eg_type2 = True
    is_small_rum = False
    if is_small_rum:
        num_epoch = 1

    n_gpu = torch.cuda.device_count()

    max_sent_len = 500
    word_embed_dim = 300
    other_embed_dim = 20
    apply_embed_dropout = True
    dist_bin_map = OrderedDict()
    min_word_count = 10
    hidden_dropout = 0.5
    visible_dropout = 0.5
    early_stop_cnt = 5
    max_coref_count = 5
    ignore_rel_list = ['None', 'NA', 'Other']
    cnn_filter_size = 500
    n_grams = [3, 4, 5]
    ngram_count = len(n_grams)
    max_path_cnt = 30

    rel_name_to_idx, rel_idx_to_name = get_class_name_idx_map(os.path.join(src_data_folder, 'relations.txt'))
    # print(rel_name_to_idx)
    QASample = recordclass("QASample", "Psg1Words Psg2Words Arg1 Arg2 Arg1Loc Arg2Loc Psg1Coref Psg2Coref Paths "
                                       "Psg1EMs Psg2EMs Psg1EMsAdjMat Psg2EMsAdjMat "
                                       "Psg1Ents Psg2Ents JointAdjMat "
                                       "Arg1EMs Arg2EMs OnlyPsg1EntEMs OnlyPsg2EntEMs BothPsgEntEMs MergedAdjMat "
                                       "Psg1Sents Psg2Sents Psg1SentsAdjMat Psg2SentsAdjMat Psg1SentsEntsAdjMat "
                                       "Psg2SentsEntsAdjMat Psg1EMsWords Psg2EMsWords EntsSents PathAdjMat Relation")

    # Train a model
    if job_mode == 'train':
        custom_print(sys.argv)
        custom_print('loading data......')

        train_data_lines = open(os.path.join(src_data_folder, 'train.json')).readlines()
        random.shuffle(train_data_lines)

        if is_small_rum:
            train_data_lines = train_data_lines[:100]

        dev_lines = train_data_lines[:int(0.1 * len(train_data_lines))]
        train_lines = train_data_lines[int(0.1 * len(train_data_lines)):]

        test_lines = open(os.path.join(src_data_folder, 'test.json')).readlines()

        if is_small_rum:
            test_lines = test_lines[:100]

        t_start = datetime.datetime.now()

        custom_print('\nloading training data...')
        train_pos_data, train_neg_data = get_data(train_lines, is_training_data=True)
        custom_print('Positive:', len(train_pos_data))
        custom_print('Total negative:', len(train_neg_data))
        random.shuffle(train_neg_data)
        train_neg_data = train_neg_data[:min(len(train_neg_data), len(train_pos_data) * neg_size)]
        custom_print('Negative:', len(train_neg_data))
        train_data = train_pos_data + train_neg_data
        custom_print('Data size:', len(train_data))

        custom_print('\nloading dev data...')
        dev_pos_data, dev_neg_data = get_data(dev_lines)
        custom_print('Positive:', len(dev_pos_data))
        custom_print('Total negative:', len(dev_neg_data))
        random.shuffle(dev_neg_data)
        dev_neg_data = dev_neg_data[:min(len(dev_neg_data), len(dev_pos_data) * neg_size)]
        custom_print('Negative:', len(dev_neg_data))
        dev_data = dev_pos_data + dev_neg_data
        custom_print('Data size:', len(dev_data))

        custom_print('\nloading test data...')
        test_pos_data, test_neg_data = get_data(test_lines)
        test_data = test_pos_data + test_neg_data
        custom_print('Positive:', len(test_pos_data))
        custom_print('Negative:', len(test_neg_data))
        t_end = datetime.datetime.now()

        custom_print('\ndata loaded:', t_end - t_start)

        custom_print('Training data size:', len(train_pos_data) + len(train_neg_data))
        custom_print('Development data size:', len(dev_data))
        custom_print('Test data size:', len(test_data))

        custom_print("\npreparing vocabulary......")
        save_vocab_file = os.path.join(trg_data_folder, 'vocab.pkl')
        word_vocab, word_embed_matrix = build_vocab(train_lines, dev_lines, test_lines,
                                                    save_vocab_file, embedding_file)

        if use_coref:
            custom_print('Using coref embeddings')
        else:
            custom_print('Coref embeddings not used')

        custom_print("Training started......")

        model_file_path = os.path.join(trg_data_folder, 'model.h5py')
        torch_train(model_name, train_data, dev_data, test_data, model_file_path)

    # Test a model
    if job_mode == 'test':
        re_th = float(sys.argv[-1])
        custom_print(sys.argv)
        custom_print('Threshold:', re_th)
        custom_print("loading word vectors......")
        vocab_file_name = os.path.join(trg_data_folder, 'vocab.pkl')
        word_vocab = load_vocab(vocab_file_name)
        model_file_path = os.path.join(trg_data_folder, 'model.h5py')
        best_model = get_model(model_name)
        custom_print(best_model)

        if torch.cuda.is_available():
            best_model.cuda()
            best_model = torch.nn.DataParallel(best_model)
        best_model.load_state_dict(torch.load(model_file_path))

        # prediction

        test_lines = open(os.path.join(src_data_folder, 'test.json')).readlines()

        if is_small_rum:
            test_lines = test_lines[:100]

        test_pos_data, test_neg_data = get_data(test_lines)
        test_data = test_pos_data + test_neg_data
        custom_print('Test data size:', len(test_data))

        test_preds = predict(test_data, best_model, model_name)
        custom_print('\nBefore thresholding:\n')
        re_f1 = get_F1(test_data, test_preds, 0.0)
        custom_print('Test RE F1:', re_f1)
        custom_print('\nAfter thresholding:\n')
        re_f1 = get_F1(test_data, test_preds, re_th, os.path.join(trg_data_folder, 'test.out'))
        custom_print('Test RE F1:', re_f1)

    logger.close()
