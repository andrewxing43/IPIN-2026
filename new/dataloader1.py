import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.transforms import ToUndirected

CHILD_EDGE_MODE = "none" 

def build_subgraph(rssi_rows, label_rows, ap_num=175):
    data = HeteroData()

    # child :all 1s as features, since we have no other info for them. The model will learn to differentiate them through edges and AP interactions
    child_x = torch.ones((5, 1), dtype=torch.float)
    data['child'].x = child_x
    
    # each graph has only one label, we can store it in the graph-level attribute for convenience
    data['label'] = torch.tensor(label_rows[0]/30, dtype=torch.float).unsqueeze(0)

    # use num_ap to set ap node count, and assign indices as features for embedding lookup
    data['ap'].num_nodes = ap_num
    data['ap'].x = torch.arange(ap_num, dtype=torch.long)

    # edge definition: child-ap and child-child
    edge_index = []
    edge_attr = []
    for child_id, row in enumerate(rssi_rows):
        for ap_id, rssi_val in enumerate(row):
            if rssi_val != -200:
                edge_index.append([child_id, ap_id])
                edge_attr.append([(rssi_val + 100) / 70])

    if edge_index:
        data[('child', 'sense', 'ap')].edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        data[('child', 'sense', 'ap')].edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        data[('child', 'sense', 'ap')].edge_index = torch.empty((2, 0), dtype=torch.long)
        data[('child', 'sense', 'ap')].edge_attr = torch.empty((0, 1), dtype=torch.float)
  
    edge_cc = []
    if CHILD_EDGE_MODE == "full":
        edge_cc = [[i, j] for i in range(5) for j in range(5) if i != j]
    elif CHILD_EDGE_MODE == "ring":
        edge_cc = [[i, (i+1)%5] for i in range(5)]
        edge_cc += [[(i+1)%5, i] for i in range(5)]
    else:
        edge_cc = []
    if edge_cc:
        edge_cc_attr = [[1.0] for _ in edge_cc]
        data[('child', 'intra', 'child')].edge_index = torch.tensor(edge_cc, dtype=torch.long).t()
        data[('child', 'intra', 'child')].edge_attr = torch.tensor(edge_cc_attr, dtype=torch.float)
    else:
        data[('child', 'intra', 'child')].edge_index = torch.empty((2, 0), dtype=torch.long)    
        data[('child', 'intra', 'child')].edge_attr = torch.empty((0, 1), dtype=torch.float)

    return ToUndirected()(data)


def load_data_and_build_dataloaders(train_csv, test_csv, batch_size=64):
   
    train_df = pd.read_csv(train_csv, header=None)
    test_df = pd.read_csv(test_csv, header=None)

    all_train = train_df.values
    all_test = test_df.values

    assert all_train.shape[0] % 5 == 0 and all_test.shape[0] % 5 == 0

    def build_graph_list(array):
        graphs = []
        for i in range(0, len(array), 5):
            rssi = array[i:i+5, 0:175]
            xy = array[i:i+5, 175:177]
            graphs.append(build_subgraph(rssi, xy))
        return graphs

    train_ids = np.arange(0, len(all_train) // 5)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=42)

    def index_group(arr, ids):
        idx = ids.repeat(5) * 5 + np.tile(np.arange(5), len(ids))
        return arr[idx]

    train_graphs = build_graph_list(index_group(all_train, train_ids))
    val_graphs = build_graph_list(index_group(all_train, val_ids))
    test_graphs = build_graph_list(all_test)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader