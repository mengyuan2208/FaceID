import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cluster_dataset import ClusterDataset
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
import itertools
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim_1)
        self.conv2 = GCNConv(hidden_dim_1, hidden_dim_2)
        self.fc_iou = nn.Linear(hidden_dim_2, output_dim)
        self.fc_iop = nn.Linear(hidden_dim_2, output_dim)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # aggregate node features into graph level features
        x = global_max_pool(x, batch)
        iou_score = self.fc_iou(x)
        iop_score = self.fc_iop(x)
        return torch.cat((iou_score, iop_score), dim=1)

def train(model, data_loader_train, data_loader_test, criterion, optimizer, ground_truth_clusters_train, ground_truth_clusters_test, epochs=100):
    model.train()
    for epoch in range(epochs):
        iou_scores = []
        clusters = [] # list of lists
        total_loss = 0
        for batch in data_loader_train:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            iop_target, iou_target = batch.y[:, 0], batch.y[:, 1]
            loss_iou = criterion(output[:, 0], iou_target)
            loss_iop = criterion(output[:, 1], iop_target)
            # loss = loss_iop + loss_iou
            loss = loss_iou
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            iou_scores.extend(output[:, 0].detach().cpu().numpy())
            for i in range(batch.num_graphs):
                mask = (batch.batch == i)
                clusters.append(batch.original_indices[mask].tolist())
        de_overlapped_clusters = de_overlap(clusters, iou_scores)
        precision, recall, fscore = calculate_pairwise_metrics(de_overlapped_clusters, ground_truth_clusters_train)
        print(f'Epoch {epoch + 1}, Train Loss: {total_loss / len(data_loader_train)}, Pairwise Precision: {precision:.4f}, Pairwise Recall: {recall:.4f}, Pairwise F-score: {fscore:.4f}')

        val_loss, val_precision, val_recall, val_fscore = validate(model, data_loader_test, criterion, ground_truth_clusters_test)
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss}, Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}, Validation F-score: {val_fscore:.4f}')
        print("")
    return iou_scores, de_overlapped_clusters

def validate(model, loader, criterion, ground_truth_clusters):
    model.eval()
    total_loss = 0
    iou_scores = []
    clusters = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            iop_target, iou_target = data.y[:, 0], data.y[:, 1]
            loss_iou = criterion(output[:, 0], iou_target)
            loss_iop = criterion(output[:, 1], iop_target)
            # loss = loss_iop + loss_iou
            loss = loss_iou
            total_loss += loss.item()
            iou_scores.extend(output[:, 0].detach().cpu().numpy())
            for i in range(data.num_graphs):
                mask = (data.batch == i)
                clusters.append(data.original_indices[mask].tolist())
    de_overlapped_clusters = de_overlap(clusters, iou_scores)
    precision, recall, fscore = calculate_pairwise_metrics(de_overlapped_clusters, ground_truth_clusters)
    return total_loss / len(loader), precision, recall, fscore

def de_overlap(clusters, iou_scores):
    iou_scores_tensor = torch.tensor(iou_scores)
    sorted_indices = torch.argsort(iou_scores_tensor, descending=True)
    sorted_clusters = [clusters[i] for i in sorted_indices]
    
    seen_vertices = set()
    de_overlapped_clusters = []

    for cluster in sorted_clusters:
        unique_vertices = [v for v in cluster if v not in seen_vertices]
        if unique_vertices:
            de_overlapped_clusters.append(unique_vertices)
            seen_vertices.update(unique_vertices)

    return de_overlapped_clusters

def calculate_pairwise_metrics(predicted_clusters, ground_truth_clusters):
    ground_truth_pairs = set()
    for cluster in ground_truth_clusters:
        ground_truth_pairs.update(list(itertools.combinations(cluster, 2)))

    predicted_pairs = set()
    for cluster in predicted_clusters:
        predicted_pairs.update(list(itertools.combinations(cluster, 2)))

    # Calculate TP, FP, FN
    tp = len(predicted_pairs & ground_truth_pairs)
    fp = len(predicted_pairs - ground_truth_pairs)
    fn = len(ground_truth_pairs - predicted_pairs)

    # Calculate precision and recall
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    fscore = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, fscore



if __name__ == "__main__":
    args_train = {
        "feat_path": "data/features/ytb_train.bin",
        "label_path": "data/labels/ytb_train.meta",
        "super_vertex_path_prefix": "./super_vertex_train"
    }

    data_instance_train = ClusterDataset(args_train)
    data_loader_train = DataLoader(data_instance_train, batch_size=32, shuffle=True)

    ground_truth_clusters_train, _ = data_instance_train.read_labels(args_train["label_path"])
    ground_truth_clusters_train = list(ground_truth_clusters_train.values())

    args_test = {
        "feat_path": "data/features/ytb_test.bin",
        "label_path": "data/labels/ytb_test.meta",
        "super_vertex_path_prefix": "./super_vertex_test"
    }

    data_instance_test = ClusterDataset(args_test)
    data_loader_test = DataLoader(data_instance_test, batch_size=32, shuffle=True)

    ground_truth_clusters_test, _ = data_instance_test.read_labels(args_test["label_path"])
    ground_truth_clusters_test = list(ground_truth_clusters_test.values())

    input_dim = 256
    hidden_dim_1 = 512
    hidden_dim_2 = 64
    output_dim = 1
    model = GCN(input_dim, hidden_dim_1, hidden_dim_2, output_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    final_iou_scores, final_clusters = train(model, data_loader_train, data_loader_test, criterion, optimizer, ground_truth_clusters_train, ground_truth_clusters_test, epochs=100)
    # print(final_clusters)

    with open('final_clusters.pkl', 'wb') as f:
        pickle.dump(final_clusters, f)

    # Save the trained model
    torch.save(model.state_dict(), 'gcn_model.pth')

    print("Model and final clusters saved successfully.")