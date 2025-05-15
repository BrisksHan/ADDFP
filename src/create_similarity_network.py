from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch


def create_side_effect_graph(frequency_matrix, graph_type = 'cosine', top_k = 10):
    #for item in frequency_matrix:
    #    print(len(item))
    frequency_matrix = np.array(frequency_matrix)
    #print(frequency_matrix.shape)
    if graph_type == 'cosine':
        similarity_matrix = cosine_graph(frequency_matrix)
        #print(similarity_matrix)
        #print(similarity_matrix.shape)
         # You can change this value to get a different number of top indices
        top_k_indices = []
        for i, row in enumerate(similarity_matrix):
            sorted_indices = np.argsort(-row)
            filtered_indices = [idx for idx in sorted_indices if idx != i and row[idx] > 0][:top_k]
            #print((len([idx for idx in sorted_indices if idx != i and row[idx] > 0])))
            top_k_indices.append(filtered_indices)

        # Convert to PyG format (COO) and ensure undirected
        edges = set()
        for i, neighbors in enumerate(top_k_indices):
            for j in neighbors:
                # Always put smaller index first to avoid duplicates
                edge = (min(i, j), max(i, j))
                edges.add(edge)

        sources = []
        targets = []

        for item in edges:
            sources.append(item[0])
            targets.append(item[1])
            sources.append(item[1])
            targets.append(item[0])
        return torch.tensor([sources, targets], dtype=torch.long)
    
    if graph_type == 'implicit':
        #bigger the better
        #implicit_matrix = np.zeros((994, 994))
        #print(frequency_matrix.shape)
        sources = []
        targets = []
        edge_attr = []
        
        max_num = 0.00001

        #new_frequency_matrix = np.zeros((994, 994))

        for i in range(994):
            for j in range(i+1,994):
                cur_edge_attr = np.dot(frequency_matrix.T[i], frequency_matrix.T[j])
                cur_edge_attr = np.log(cur_edge_attr)
                if cur_edge_attr > max_num:
                    max_num = cur_edge_attr

        for i in range(994):
            for j in range(i+1,994):
                cur_edge_attr = np.dot(frequency_matrix.T[i], frequency_matrix.T[j])/max_num
                if cur_edge_attr > 0.4:
                    sources.append(i)
                    targets.append(j)
                    edge_attr.append(cur_edge_attr)
                    sources.append(j)
                    targets.append(i)
                    edge_attr.append(cur_edge_attr)

                #if cur_edge_attr > 0.05:
                #    sources.append(i)
                #    targets.append(j)
                #    edge_attr.append(cur_edge_attr)
                #    sources.append(j)
                #    targets.append(i)
                #    edge_attr.append(cur_edge_attr)
        print('edge num:', len(sources)/2)
        return torch.tensor([sources, targets], dtype=torch.long), torch.tensor(edge_attr, dtype=torch.float)

    
def cosine_graph(frequency_matrix):
    similarity_matrix = cosine_similarity(frequency_matrix.T)
    return similarity_matrix

