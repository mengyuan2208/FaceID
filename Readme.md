# Facial Clustering Model Based on Graph Convolutional Network

This repository contains code for a facial clustering model using a Graph Convolutional Network (GCN). The model is based on the paper [Learning to Cluster Faces on an Affinity Graph](https://arxiv.org/abs/1904.02749). 

## Model Overview

The model consists of two main components:

1. **Cluster Component**: This component proposes possible clusters by iteratively merging small clusters from an affinity graph built on extracted features.
2. **Detection Component**: This component uses a GCN to predict the Intersection over Union (IoU) at the graph level. The model ranks potential clusters based on predicted IoU and performs de-overlap to form the final clusters.

Model Performance

- **Validation Precision**: 0.8116
- **Validation Recall**: 0.6289
- **Validation F-Score**: 0.7087

## Requirements

Python (Version >= 3.9 recommended)

PyTorch

Torch Geometric

faiss

Torchvision

Pillow

Scipy

NumPy

You can install the required Python packages using `pip`:

```
pip install torch torch-geometric torchvision pillow scipy numpy faiss-cpu
```

## How to use the model

### 1. Preprocessing: Extract Features

Before clustering, you need to extract features from the input images.

Run the preprocessing step with:

```
sh run_preprocess.sh <filelist>
```

**Input**: A text file (`filelist`) listing the paths of cropped and aligned face images.

**Output**: Features are extracted using a pre-trained ResNet model and saved in `feature.bin`.

### 2. Cluster Proposals: Generate Proposals

After feature extraction, generate cluster proposals:

```
sh run_proposal.sh
```

**Output**: The script iteratively generates and merges cluster proposals, saving the final clusters in the `./super_vertex` folder.

### 3. Prediction: Final Clustering

Once proposals are generated, use the trained GCN model to rank and de-overlap the clusters:

``` 
sh run_predict.sh
```

**Output**: The final de-overlapped clusters are saved in `clusters.txt`. Each line represents a cluster, with IDs corresponding to the image indices in the `filelist`.

### 4. Clean Up: Remove Intermediate Files

To remove intermediate files generated during the process (e.g., feature files, super vertex proposals), run:

```
sh run_clean.sh
```

This will clean up the `__pycache__` folders, `feat.bin`, and `super_vertex` directories.
