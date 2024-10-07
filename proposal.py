import numpy as np
import faiss
import os
from scipy.sparse import csr_matrix
import argparse

class Proposal(object):
    def __init__(self, args):
        self.feat_dim = 256
        self.k = args.k
        self.threshold = args.threshold
        self.maxsize = args.maxsize
        self.threshold_step = args.threshold_step
        self.iter_num = args.iter_num
        self.feat_path = args.feat_path
        self.feat = None
        self.super_vertex_path_prefix = args.super_vertex_path_prefix
        self.num = 0

    def read_feat(self):
        # read features from self.feat_path and normalizes it
        self.feat = np.fromfile(self.feat_path, dtype=np.float32, count=-1)
        self.feat = self.feat.reshape(-1, self.feat_dim)
        self.feat /= np.linalg.norm(self.feat, axis=1).reshape(-1, 1)
        self.num = self.feat.shape[0]
        print("Features normalized, size:", self.feat.shape)

    def build_affinity(self, feat):
        # build affinity graph using faiss, return the index and distance matrix (n * k)
        index_cpu = faiss.IndexFlatIP(self.feat_dim)
        # res = faiss.StandardGpuResources()
        # index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        # index_gpu.add(feat)
        index_cpu.add(feat)

        D, I = index_cpu.search(feat, self.k + 1)
        I_excluded = I[:, 1:]
        D_excluded = D[:, 1:]
        print("Build affinity graph, index matrix size: ", I_excluded.shape, ", distance matrix size: ", D_excluded.shape)
        return I_excluded, D_excluded

    def find_super_vertex(self, I, D, max_iter=100):
        # find super vectices iteratively until no remaining nodes in R, return a list of sets
        S = []
        R = set(range(len(I)))
        threshold = self.threshold
        C, R = self.find_super_vertices(I, D, threshold, self.maxsize, R)
        print("Threshold: ", threshold, ", size of newly find super vertex: ", len(C), ", number of remaining nodes: ", len(R))
        S.extend(C)

        iter = 0

        while R and iter < max_iter:
            threshold = threshold + (1 - threshold) * self.threshold_step
            C, R = self.find_super_vertices(I, D, threshold, self.maxsize, R)
            print("Threshold: ", threshold, ", size of newly find super vertex: ", len(C), ", number of remaining nodes: ", len(R))
            S.extend(C)
            iter += 1
        
        if R:
            S.append(list(R))
        
        print("Build super vertex, size of super vertex: ", len(S), ", size of sample: ", len(I))

        return S

    def find_super_vertices(self, I, D, e_tau, smax, R):
        # find super vertex in the filtered affinity graph and keep the remaining nodes of the super vertex larger then smax in R
        # return the super veretx (list of sets) and the remaining nodes (set)
        R_original = R
        A_prime = self.prune_edge(I, D, e_tau, R)
        X = self.find_connected_components(A_prime)
        C = [component for component in X if len(component) < smax]
        remaining_sets = [component for component in X if len(component) >= smax]
        R = set().union(*remaining_sets)
        assert R.issubset(R_original)
        return C, R

    def prune_edge(self, I, D, e_tau, R):
        # only keep edges in the affinity graph with similarity larger than e_tau and nodes in R
        # return an affinity graph (n * n)
        n = len(D)
        row_indices = []
        col_indices = []
        data = []
        # A_prime = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(D.shape[1]):
                if D[i, j] > e_tau and i in R and I[i, j] in R:
                        row_indices.append(i)
                        col_indices.append(I[i, j])
                        data.append(True)
                        # A_prime[i, I[i, j]] = True
                        # A_prime[I[i, j], i] = True
        A_prime = csr_matrix((data, (row_indices, col_indices)), shape=(n, n), dtype = bool)
        A_prime = A_prime + A_prime.T
        return A_prime

    def find_connected_components(self, A_prime):
        # Find connected components using DFS
        n = A_prime.shape[0]
        visited = [False] * n
        components = []

        def dfs(node):
            stack = [node]
            component = set()
            while stack:
                u = stack.pop()
                if not visited[u]:
                    visited[u] = True
                    component.add(u)
                    # stack.extend(v for v in range(n) if A_prime[u, v] and not visited[v])
                    for v in A_prime.indices[A_prime.indptr[u]:A_prime.indptr[u+1]]:
                        if not visited[v]:
                            stack.append(v)
            return component

        for i in range(n):
            if not visited[i] and A_prime[i].count_nonzero() > 0:
            # if not visited[i] and A_prime[i].any():
                components.append(dfs(i))

        return components
    
    def generate_iter_proposals(self):
        for i in range(self.iter_num):
            print("Iteration ", i)
            if i == 0:
                self.read_feat()
                I, D = self.build_affinity(self.feat)
                S = self.find_super_vertex(I, D)

                all_nodes = set(range(len(I)))
                clustered_nodes = set()
                for cluster in S:
                    clustered_nodes.update(cluster)
                unclustered_nodes = all_nodes - clustered_nodes
                for node in unclustered_nodes:
                    S.append([node])

                labels = {} # id of vertex: id of super vertex
                for index, cluster in enumerate(S):
                    for v in cluster:
                        labels[v] = index
                sv_folder_path = os.path.join(
                    self.super_vertex_path_prefix, 'k_{}_threshold_{}_step_{}_minsz_{}_maxsz_{}_iter_{}'.format(
                        self.k, self.threshold, self.threshold_step, 1, self.maxsize, i))
                sv_labels_path = os.path.join(sv_folder_path, 'labels')
                self.save_labels(sv_labels_path, labels, self.num)

                S_filtered = [component for component in S if len(component) >= 1]
                self.save_proposals(sv_folder_path, S_filtered, I, D)
            else:
                if i == 1:
                    sv_folder_path_last_iter = os.path.join(
                        self.super_vertex_path_prefix, 'k_{}_threshold_{}_step_{}_minsz_{}_maxsz_{}_iter_{}'.format(
                    self.k, self.threshold, self.threshold_step, 1, self.maxsize, i - 1))
                else:
                    sv_folder_path_last_iter = os.path.join(
                        self.super_vertex_path_prefix, 'k_{}_threshold_{}_step_{}_minsz_{}_maxsz_{}_iter_{}'.format(
                    self.k, self.threshold, self.threshold_step, 2, self.maxsize, i - 1))
                sv_labels_path_last_iter = os.path.join(sv_folder_path_last_iter, 'labels')
                print('read sv_clusters from {}'.format(sv_labels_path_last_iter))
                sv_lb2idxs, sv_idx2lb = self.read_labels(sv_labels_path_last_iter)
                sv_clusters = [idxs for _, idxs in sv_lb2idxs.items()]
                feat = np.array([self.feat[list(c), :].mean(axis=0) for c in sv_clusters])
                print('Average feature of super vertices:', feat.shape)
                I_new, D_new = self.build_affinity(feat)
                S = self.find_super_vertex(I_new, D_new)

                all_cluster_indices = set(range(len(I_new)))
                assigned_clusters = set(cluster_idx for super_vertex in S for cluster_idx in super_vertex)
                unassigned_clusters = all_cluster_indices - assigned_clusters
                for cluster_idx in unassigned_clusters:
                    S.append([cluster_idx])
                
                clusters = []
                for super_vertex in S:
                    merged_cluster = []
                    for cluster_idx in super_vertex:
                        merged_cluster.extend(sv_clusters[cluster_idx])
                    clusters.append(merged_cluster)
                assert len(clusters) == len(S)
                
                labels = {} # id of vertex: id of super vertex
                for index, cluster in enumerate(clusters):
                    for v in cluster:
                        labels[v] = index
                sv_folder_path = os.path.join(
                    self.super_vertex_path_prefix, 'k_{}_threshold_{}_step_{}_minsz_{}_maxsz_{}_iter_{}'.format(
                        self.k, self.threshold, self.threshold_step, 2, self.maxsize, i))
                sv_labels_path = os.path.join(sv_folder_path, 'labels')
                self.save_labels(sv_labels_path, labels, self.num)

                S_filtered = [sv for sv in S if len(sv) >= 2]
                clusters_filtered = []
                for super_vertex in S_filtered:
                    merged_cluster = []
                    for cluster_idx in super_vertex:
                        merged_cluster.extend(sv_clusters[cluster_idx])
                    clusters_filtered.append(merged_cluster)
                assert len(clusters_filtered) == len(S_filtered)
                self.save_proposals(sv_folder_path, clusters_filtered, I, D)
                


    def read_labels(self, label_path):
        lb2idxs = {}
        idx2lb = {}
        with open(label_path) as f:
            for idx, x in enumerate(f.readlines()):
                lb = int(x.strip())
                if lb not in lb2idxs:
                    lb2idxs[lb] = []
                lb2idxs[lb] += [idx]
                idx2lb[idx] = lb
        print("Loaded {} labels belonging to {} samples.".format(len(lb2idxs), len(idx2lb)))
        return lb2idxs, idx2lb

    def save_labels(self, sv_labels_path, labels, num):
        print('save labels to', sv_labels_path)
        os.makedirs(os.path.dirname(sv_labels_path), exist_ok=True)
        with open(sv_labels_path, 'w') as of:
            for idx in range(num):
                of.write(str(labels[idx]) + '\n')
    
    def save_proposals(self, sv_proposals_path, S, I, D):
        print('save nodes and edges to', sv_proposals_path)
        for lb, nodes in enumerate(S):
            sv_proposals_path_node = os.path.join(sv_proposals_path, '{}_node.npz'.format(lb))
            sv_proposals_path_edge = os.path.join(sv_proposals_path, '{}_edge.npz'.format(lb))
            os.makedirs(os.path.dirname(sv_proposals_path_node), exist_ok=True)
            os.makedirs(os.path.dirname(sv_proposals_path_edge), exist_ok=True)
            edges = []
            for idx in nodes:
                neighbors = I[idx]
                dists = D[idx]
                for n, dist in zip(neighbors, dists):
                    if n not in nodes:
                        continue
                    edges.append([idx, n, 1 - dist])
            # save to npz file
            np.savez_compressed(sv_proposals_path_node, list(nodes))
            np.savez_compressed(sv_proposals_path_edge, edges)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Proposal Generation")
    parser.add_argument("--k", type=int, default=160, help="Number of nearest neighbors")
    parser.add_argument("--threshold", type=float, default=0.70, help="Initial threshold for affinity graph pruning")
    parser.add_argument("--maxsize", type=int, default=300, help="Maximum size of super vertex")
    parser.add_argument("--threshold_step", type=float, default=0.05, help="Threshold step for each iteration")
    parser.add_argument("--iter_num", type=int, default=3, help="Number of iterations")
    parser.add_argument("--feat_path", type=str, default="feat.bin", help="File of extracted features")
    parser.add_argument("--super_vertex_path_prefix", type=str, default="./super_vertex", help="Place to store the proposals")

    args = parser.parse_args()

    proposal_instance = Proposal(args)
    proposal_instance.generate_iter_proposals()