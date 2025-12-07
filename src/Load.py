import os
import random
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
from torch.utils.data import TensorDataset, DataLoader
from itertools import islice
from src.generate_graph import get_theoretical_network

def Undirected(inter, network_layer):
    # Undirected_graph
    inter = inter.drop_duplicates(subset=['left', 'right']).reset_index(drop=True)
    inter_bi = pd.concat([
        inter,
        inter.rename(columns={'left': 'right', 'right': 'left'})
    ], ignore_index=True)
    inter_bi = inter_bi.drop_duplicates().reset_index(drop=True)

    inter_bi['layer'] = network_layer
    inter_bi['exist'] = 1
    df = inter_bi[['layer', 'left', 'right', 'exist']]
    num_unedges = len(inter_bi)
    return df, num_unedges


def AutoNegSampler(pos_df, each_layer_infor, num_layers, max_neg_limit=200000, balance_neg_ratio=100):
    # Full sampling is prioritized;

    total_all_neg = 0
    layer_nodes_set = []
    for layer in range(num_layers):
        N = each_layer_infor[layer]['num_nodes']
        E = each_layer_infor[layer]['num_bi_edges']
        layer_all_neg = N * (N - 1) - E
        each_layer_infor[layer]['num_all_bi_neg'] = layer_all_neg
        total_all_neg += layer_all_neg
        print(f"[Layer {layer}]. N: {N}. bi_E: {E}. all_bi_neg: {layer_all_neg}.")
        layer_nodes = set(each_layer_infor[layer]['nodes'])
        layer_nodes_set.append(layer_nodes)


    n_train = 0
    n_valid = 0
    n_test = 0
    layer_num_neg = []
    use_all_sampling = total_all_neg <= max_neg_limit
    if use_all_sampling:
        print(f"All sampling: {total_all_neg}.")
    else:
        print("train num_pos is sufficient. 1:1 balance for training set.")
        # When the training set is large enough, a 1:1 undersampling of the training set is sufficient;
        # negative samples can be sampled at a 1:1 ratio for training the model.
        # balance for trainset
        # still imbalanced sampling for the evaluation phase
        grouped_count = pos_df.groupby('layer')['x_label'].value_counts()
        for layer in range(num_layers):
            num_pos_valid = grouped_count.loc[layer, 1]
            num_pos_test = grouped_count.loc[layer, 2]
            all_ratio = min(each_layer_infor[layer]['num_all_bi_neg'] / each_layer_infor[layer]['num_bi_edges'], balance_neg_ratio)
            num_neg_train = grouped_count.loc[layer, 0]
            num_neg_valid = num_pos_valid * all_ratio
            num_neg_test = num_pos_test * all_ratio
            n_train += int(num_neg_train)
            n_valid += int(num_neg_valid)
            n_test += int(num_neg_test)
            layer_n_neg = int(num_neg_train + num_neg_valid + num_neg_test)
            print(f"[Layer {layer}]. valid/test sample ratio:{all_ratio}. sampling: {layer_n_neg}.")
            node_num_neg = layer_n_neg // each_layer_infor[layer]['num_nodes']
            layer_num_neg.append(node_num_neg)

    samples = []
    grouped = pos_df.groupby(['layer', 'left'])
    for (layer, node), pos_inters in grouped:
        pos_list = set(pos_inters['right'].tolist())
        forbid = pos_list | {node}
        candidates = list(layer_nodes_set[layer] - forbid)
        if use_all_sampling:
            neg_nodes = candidates
        else:
            num_neg_node = layer_num_neg[layer]
            if len(candidates) >= num_neg_node:
                neg_nodes = random.sample(candidates, num_neg_node)
            else:
                neg_nodes = random.choices(candidates, k=num_neg_node)
        samples.extend([[layer, node, n, 0] for n in neg_nodes])


    n = len(samples)
    if use_all_sampling:
        n_train = int(0.8 * n)
        n_valid = int(0.1 * n)
        n_test = n - n_train - n_valid
    print(f"neg sampling: {n}. neg_train: {n_train}. neg_valid: {n_valid}. neg_test: {n_test}.")
    idx = np.random.permutation(n)
    x_label = np.zeros(n, dtype=np.int8)
    x_label[idx[n_train:n_train + n_valid]] = 1
    x_label[idx[n_train + n_valid:]] = 2
    neg_data = pd.DataFrame(samples, columns=["layer", "left", "right", "exist"])
    neg_data['x_label'] = x_label
    data = pd.concat([pos_df, neg_data], axis=0, ignore_index=True)
    data = data.sample(frac=1).reset_index(drop=True)
    print('-----------------------------------')
    return data

def dynamic_balance_resample(train_infor, seed, limit=10000):
    train_label = train_infor[:, 3]
    counter = Counter(train_label)
    num_neg = counter[0]
    num_pos = counter[1]
    imb_ratio = num_neg / num_pos

    # Case 1: Sufficient number of positive samples for training → Undersampling
    if num_pos >= limit:
        print(f"train num_pos is sufficient.")
        # rus = RandomUnderSampler(
        #     sampling_strategy={0: num_pos, 1: num_pos},
        #     random_state=seed
        # )
        # train_infor, train_label = rus.fit_resample(train_infor, train_label)
        # print("train resampling: ", sorted(Counter(train_label).items()))
        return train_infor

    # Case 2: Insufficient positive samples → (under-sampling + over-sampling)
    print(f"Original train: num_neg={num_neg}, num_pos={num_pos}, ratio≈{imb_ratio:.2f}")
    K = int(max(3, min(imb_ratio // 10, 20)))
    balance_num = int(num_pos * K)
    balance_num = min(balance_num, num_neg)
    print(f"Dynamic K = {K}, balance_num = {balance_num}")
    steps = [
        ('under', RandomUnderSampler(sampling_strategy={0: balance_num, 1: num_pos}, random_state=seed)),
        ('over', RandomOverSampler(sampling_strategy=1.0, random_state=seed))
    ]
    pipeline = Pipeline(steps)
    train_infor, train_label = pipeline.fit_resample(train_infor, train_label)
    print("train resampling: ", sorted(Counter(train_label).items()))
    print('-----------------------------------')
    return train_infor


def split_dataset(df, each_layer_infor, num_layers, random_state):
    n = len(df)
    idx = np.random.RandomState(random_state).permutation(n)
    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)
    n_test = n - n_train - n_valid
    print(f"pos sampling: {n}. pos_train: {n_train}. pos_valid: {n_valid}. pos_test: {n_test}.")
    x_label = np.zeros(n, dtype=np.int8)
    x_label[idx[n_train:n_train + n_valid]] = 1
    x_label[idx[n_train + n_valid:]] = 2
    df['x_label'] = x_label
    df_final = AutoNegSampler(df, each_layer_infor, num_layers)
    return df_final


def split_infor(df):
    train_infor = df[df['x_label'] == 0].reset_index(drop=True)
    valid_infor = df[df['x_label'] == 1].reset_index(drop=True)
    test_infor = df[df['x_label'] == 2].reset_index(drop=True)
    return train_infor, valid_infor, test_infor


def get_loader(infor, batch_size):
    network = torch.LongTensor(infor[:, 0])
    leftnode = torch.LongTensor(infor[:, 1])
    rightnode = torch.LongTensor(infor[:, 2])
    link = torch.LongTensor(infor[:, 3])
    data_set = TensorDataset(network, leftnode, rightnode, link)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    return data_loader

def layerwise_gcndata_load(inters, all_nodes):
    pos_edge = np.array(inters[['left', 'right']]).tolist()
    g = nx.Graph()
    g.add_nodes_from(all_nodes)
    g.add_edges_from(pos_edge)
    # adj = nx.to_scipy_sparse_matrix(g, nodelist=all_nodes, dtype=int, format='coo')  # network2 python3.7
    adj = nx.to_scipy_sparse_array(g, nodelist=all_nodes, dtype=int, format='coo')  # network3 python3.8

    edge_index = torch.LongTensor(np.vstack((adj.row, adj.col)))
    x = torch.unsqueeze(torch.FloatTensor(all_nodes), 1)
    layerwise_gcn_data = Data(x=x, edge_index=edge_index)
    return layerwise_gcn_data

def gcn_load(df, all_nodes):
    gcn_datas = []
    result = df[df['exist'] == 1].groupby('layer')[['left', 'right']]
    for layer, group_inter in result:
        gcn_datas.append(layerwise_gcndata_load(group_inter, all_nodes))
    return gcn_datas



def load_data_support(dataset, batch_size, set_seed):
    datadir = 'data/' + dataset + '_data/'
    layerfiles = os.listdir(datadir)
    num_layers = len(layerfiles)
    change = []
    whole_edges_num = 0

    each_layer_infor = {}
    for i in range(num_layers):
        now_layer = datadir + dataset + str(i+1) + '.txt'
        now_inter = pd.read_csv(now_layer, sep=' ', header=None)
        now_nodes = list(set(np.array(now_inter).reshape(-1)))
        print('-----------------------------------')
        print('Nodes of layer ' + str(i + 1) + ": " + str(len(now_nodes)))
        print('Edges of layer ' + str(i + 1) + ": " + str(now_inter.shape[0]))
        each_layer_infor[i] = {"num_nodes": len(now_nodes), "num_inter": now_inter.shape[0]}
        whole_edges_num += now_inter.shape[0]
        change += now_nodes
    change = list(set(change))
    change_dict = {}
    for i in range(len(change)):
        change_dict[change[i]] = i
    whole_nodes = list(change_dict.values())
    # for k, v in islice(change_dict.items(), 10):
    #     print(k, v)
    print('-----------------------------------')
    print('Nodes of all layers: ', len(whole_nodes))
    print('Edges of all layers: ', whole_edges_num)
    print('-----------------------------------')

    data = pd.DataFrame()
    mapped_inter = []
    for i in range(num_layers):
        now_layer = datadir + dataset + str(i+1) + '.txt'
        now_inter = pd.read_csv(now_layer, sep=' ', header=None, names=['left', 'right'])
        now_inter['left'] = now_inter['left'].map(change_dict)
        now_inter['right'] = now_inter['right'].map(change_dict)
        now_nodes = list(set(np.array(now_inter).reshape(-1)))
        each_layer_infor[i]['nodes'] = now_nodes
        layer_result, num_unedges = Undirected(now_inter, i)
        each_layer_infor[i]['num_bi_edges'] = num_unedges
        data = pd.concat([data, layer_result], axis=0).reset_index(drop=True)
        mapped_inter.append(now_inter)

    data = split_dataset(data, each_layer_infor, num_layers, set_seed)
    train_infor, valid_infor, test_infor = split_infor(data)
    gcn_data = gcn_load(train_infor, whole_nodes)

    train_infor = train_infor.to_numpy()
    valid_infor = valid_infor.to_numpy()
    test_infor = test_infor.to_numpy()
    print("train counter: ", sorted(Counter(train_infor[:, 3]).items()))
    print("valid counter: ", sorted(Counter(valid_infor[:, 3]).items()))
    print("test counter: ", sorted(Counter(test_infor[:, 3]).items()))
    # training set
    train_infor = dynamic_balance_resample(train_infor, set_seed)
    train_loader = get_loader(train_infor, batch_size)
    valid_loader = get_loader(valid_infor, batch_size)
    test_loader = get_loader(test_infor, batch_size)
    return train_loader, valid_loader, test_loader, gcn_data, num_layers

def pro_data_CLGC(input_layers, set_seed):

    theoretical_networks = ['small_world', 'scale_free', 'random_graph']
    if set(input_layers) & set(theoretical_networks):
        assert len(input_layers) == 2, "analysis of a single network and a theoretical network"
        another_networks = [name for name in input_layers if name not in theoretical_networks]
        dataset = another_networks[0].split("_")
        dataset, read_layer = dataset[0], dataset[1]
        layer_file = 'data/' + dataset + '_data/' + dataset + read_layer + '.txt'
        layer_inter = pd.read_csv(layer_file, sep=' ', header=None)
        layer_nodes = list(set(np.array(layer_inter).reshape(-1)))
        th_n = len(layer_nodes)
        th_m = layer_inter.shape[0]

    network_infor = {}
    change_dicts = {}
    for i in range(len(input_layers)):
        name = input_layers[i]
        network_infor[name] = {}
        network_infor[name]['lay_index'] = i
        network_infor[i] = {}
        if name in ['small_world', 'scale_free', 'random_graph']:
            dataset = name
            layer_inter, gener_path = get_theoretical_network(name, th_n, th_m)
            layer_nodes = list(set(np.array(layer_inter).reshape(-1)))
            network_infor[i]['file_path'] = gener_path
        else:
            dataset = name.split("_")
            dataset, read_layer = dataset[0], dataset[1]
            layer_file = 'data/' + dataset + '_data/' + dataset + read_layer + '.txt'
            network_infor[i]['file_path'] = layer_file
            layer_inter = pd.read_csv(layer_file, sep=' ', header=None)
            layer_nodes = list(set(np.array(layer_inter).reshape(-1)))
        print('-----------------------------------')
        print(f"Nodes of {name}: {len(layer_nodes)}")
        print(f"Edges of {name}: {layer_inter.shape[0]}")
        network_infor[i]['dataset'] = dataset
        network_infor[i]["num_nodes"] = len(layer_nodes)
        network_infor[i]["num_inter"] = layer_inter.shape[0]
        print('-----------------------------------')
        if dataset not in change_dicts:
            change_dicts[dataset] = {}
        for node in layer_nodes:
            if node not in change_dicts[dataset]:
                change_dicts[dataset][node] = len(change_dicts[dataset])

    data = pd.DataFrame()
    for i in range(len(input_layers)):
        dataset = network_infor[i]['dataset']
        layer_inter = pd.read_csv(network_infor[i]['file_path'], sep=' ', header=None, names=['left', 'right'])
        layer_inter['left'] = layer_inter['left'].map(change_dicts[dataset])
        layer_inter['right'] = layer_inter['right'].map(change_dicts[dataset])
        now_nodes = list(set(np.array(layer_inter).reshape(-1)))
        layer_result, num_unedges = Undirected(layer_inter, i)
        network_infor[i]['nodes'] = now_nodes
        network_infor[i]['num_bi_edges'] = num_unedges
        data = pd.concat([data, layer_result], axis=0).reset_index(drop=True)

    data = split_dataset(data, network_infor, len(input_layers), set_seed)
    group = data.groupby('layer')[['layer', 'left', 'right', 'exist', 'x_label']]
    for layer, group_inter in group:
        train_df = group_inter[group_inter['x_label'] == 0]
        dataset = network_infor[layer]['dataset']
        dataset_nodes = list(change_dicts[dataset].values())
        train_pos_df = train_df[train_df['exist'] == 1]
        network_infor[layer]["layerwise_gcn_data"] = layerwise_gcndata_load(train_pos_df, dataset_nodes)
        train_df = train_df.to_numpy()
        train_df = dynamic_balance_resample(train_df, set_seed)
        network_infor[layer]["train"] = train_df
        network_infor[layer]["valid"] = group_inter[group_inter['x_label'] == 1].to_numpy()
        network_infor[layer]["test"] = group_inter[group_inter['x_label'] == 2].to_numpy()
    return network_infor

