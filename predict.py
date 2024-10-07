# to be modified...
import torch
from torch_geometric.loader import DataLoader
from cluster_dataset import ClusterDataset
from model import GCN
from model import de_overlap

# Load the saved model
input_dim = 256
hidden_dim_1 = 512
hidden_dim_2 = 64
output_dim = 1

model = GCN(input_dim, hidden_dim_1, hidden_dim_2, output_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'gcn_model.pth'
if device.type == 'cpu':
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load(model_path))
model.eval()
model = model.to(device)

args = {
    "feat_path": "feat.bin",
    "super_vertex_path_prefix": "./super_vertex"
}

data_instance = ClusterDataset(args)
data_loader = DataLoader(data_instance, batch_size=32, shuffle=False)

model.eval()
predicted_iou_scores = []
predicted_clusters = []

with torch.no_grad(): 
    for batch in data_loader:
        batch = batch.to(device)
        outputs = model(batch)
        iou_scores = outputs[:, 0]
        
        predicted_iou_scores.extend(iou_scores.cpu().numpy())
        
        for i in range(batch.num_graphs):
            mask = (batch.batch == i)
            predicted_clusters.append(batch.original_indices[mask].tolist())

de_overlapped_clusters = de_overlap(predicted_clusters, predicted_iou_scores)
with open('clusters.txt', 'w') as f:
    for cluster in de_overlapped_clusters:
        f.write(" ".join(map(str, cluster)) + "\n")

print("De-overlapped clusters saved successfully to clusters.txt.")