import numpy as np
import os
import glob
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Dataset


class ClusterDataset(Dataset):
    def __init__(self, args):
        """
        Initialize the dataset class with feature paths and other configurations.
        Inherits from torch.utils.data.Dataset.
        """
        self.proposals = []   # Stores node and edge files (proposals)
        self.super_vertex_path_prefix = args["super_vertex_path_prefix"]
        self.feat_path = args["feat_path"]
        # self.label_path = args["label_path"]
        self.feat = None
        self.feat_dim = 256
        # self.lb2idxs = {} # {0: [0, 1, 2, ...], 1: [5]}, label to index mapping
        # self.idx2lb = {} # {0: 0, 1: 0, ...}, index to label mapping
        self.proposals = []

        self.load_all_data()

    def read_feat(self, feat_path):
        """
        Read features from the binary file and normalize them.
        """
        feat = np.fromfile(feat_path, dtype=np.float32, count=-1)
        feat = feat.reshape(-1, self.feat_dim)
        feat /= np.linalg.norm(feat, axis=1).reshape(-1, 1)
        print("Features normalized, size:", feat.shape)
        return feat

    # def read_labels(self, label_path):
    #     """
    #     Read the labels and create mappings between labels and node indices.
    #     """
    #     lb2idxs = {}
    #     idx2lb = {}
    #     with open(label_path) as f:
    #         for idx, x in enumerate(f.readlines()):
    #             lb = int(x.strip())
    #             if lb not in lb2idxs:
    #                 lb2idxs[lb] = []
    #             lb2idxs[lb] += [idx]
    #             idx2lb[idx] = lb
    #     print("Loaded {} labels belonging to {} classes.".format(len(lb2idxs), len(idx2lb)))
    #     return lb2idxs, idx2lb
    
    def load_all_data(self):
        """
        Load features, labels, and proposals (node and edge files).
        """
        self.feat = self.read_feat(self.feat_path)
        # self.lb2idxs, self.idx2lb = self.read_labels(self.label_path)

        unique_nodes = set()
        for root, dirs, files in os.walk(self.super_vertex_path_prefix):
            node_files = sorted(glob.glob(os.path.join(root, '*_node.npz')))
            edge_files = sorted(glob.glob(os.path.join(root, '*_edge.npz')))
            for fn_node, fn_edge in zip(node_files, edge_files):
                self.proposals.append([fn_node, fn_edge])
                node_data = np.load(fn_node, allow_pickle=True)["arr_0"]
                unique_nodes.update(node_data)
        print('loaded {} proposals'.format(len(self.proposals)))
        print('Total number of unique nodes: {}'.format(len(unique_nodes)))


    # def compute_iop_iou(self, fn_node):
    #     """
    #     Compute Intersection over Proposal (IoP) and Intersection over Union (IoU) for the graph nodes.
    #     """
    #     node = np.load(fn_node, allow_pickle=True)["arr_0"]
    #     lb2cnt = {}
    #     for idx in node:
    #         lb = self.idx2lb[idx]
    #         if lb not in lb2cnt:
    #             lb2cnt[lb] = 0
    #         lb2cnt[lb] += 1
    #     ground_truth_label = max(lb2cnt, key=lb2cnt.get)
    #     ground_truth_node = self.lb2idxs[ground_truth_label]
    #     intersection = set(node).intersection(set(ground_truth_node))
    #     union = set(node).union(set(ground_truth_node))

    #     iop = len(intersection) / len(node)
    #     iou = len(intersection) / len(union)
    #     return iop, iou
    
    def build_graph(self, fn_node, fn_edge):
        """
        Build a graph from node and edge files.
        """
        # iop, iou = self.compute_iop_iou(fn_node)
        features, node_mapping, original_indices = self.build_feat(fn_node)
        edge_index, edge_attr = self.build_edge(fn_edge, node_mapping)
        # return iop, iou, features, edge_index, edge_attr, original_indices
        return features, edge_index, edge_attr, original_indices
    
    def build_feat(self, fn_node):
        """
        Build features for the graph nodes.
        """
        node = np.load(fn_node, allow_pickle=True)["arr_0"]
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(node)}
        original_indices = node
        features = self.feat[node, : ]
        return features, node_mapping, original_indices
        
    def build_edge(self, fn_edge, node_mapping):
        """
        Build edges for the graph using node mapping.
        """
        edge = np.load(fn_edge, allow_pickle=True)["arr_0"]
        if edge.size == 0:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, 1), dtype=np.float32)
        else:
            edge_index = np.array([[node_mapping[old_idx] for old_idx in edge[:, 0]],
                                [node_mapping[old_idx] for old_idx in edge[:, 1]]], dtype=np.int64)
            edge_attr = edge[:, 2:]
        return edge_index, edge_attr
    
    def __len__(self):
        """
        Return the number of proposals (graphs) in the dataset.
        """
        return len(self.proposals)
    
    def __getitem__(self, idx):
        """
        Return a single graph (Data) object for the given index.
        """
        fn_node, fn_edge = self.proposals[idx]
        features, edge_index, edge_attr, original_indices = self.build_graph(fn_node, fn_edge)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        x = torch.tensor(features, dtype=torch.float)
        # y = torch.tensor([iop, iou], dtype=torch.float).unsqueeze(0)
        original_indices = torch.tensor(original_indices, dtype=torch.long)

        # Return the graph as a PyTorch Geometric Data object
        # proposal_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, original_indices=original_indices)
        proposal_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, original_indices=original_indices)
        return proposal_data

if __name__ == "__main__":
    args = {
        "feat_path": "./data/features/ytb_train.bin",
        # "label_path": "./data/labels/ytb_train.meta",
        "super_vertex_path_prefix": "./super_vertex"
    }
    dataset_instance = ClusterDataset(args)
    data_loader = DataLoader(dataset_instance, batch_size=32, shuffle=True)

    for batch in data_loader:
        print(batch)