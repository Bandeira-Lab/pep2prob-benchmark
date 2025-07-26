from collections import defaultdict
import collections
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from utils import *

INPUT_DIT = "./dataset"
prefix_suffix_low_dist_set_dict_path = os.path.join(INPUT_DIT, "prefix_suffix_low_dist_set_dict.pkl")
test_ratio = 0.2
max_low_dist = 2
test_index_set = set()

meta_info_all = pd.read_csv(os.path.join(INPUT_DIT, "precursor_info_org.tsv"), sep="\t")
print("meta_info_all.shape:", meta_info_all.shape)
data_matrix = np.load(os.path.join(INPUT_DIT, "matrix_mz_prob_mean_var.npy"))
print("data_matrix.shape:", data_matrix.shape)
meta_info_all['original_precursor_index'] = meta_info_all['precursor_index'].copy()
meta_info_all['precursor_index'] = [i for i in range(meta_info_all.shape[0])]
meta_info_all.to_csv(os.path.join(INPUT_DIT, "precursor_info.tsv"), sep="\t", index=False)

meta_info_all.sort_values(by=["sequence_length"], inplace=True)
meta_info_all = meta_info_all[meta_info_all["sequence_length"] > 6]

# get the length of the sequences counter
length_counter = collections.Counter(meta_info_all["sequence_length"])
plt.figure(figsize=(5, 5))
plt.bar(length_counter.keys(), length_counter.values())
plt.xlabel("Length of sequences")
plt.ylabel("Number of sequences")
plt.title("Distribution of length of sequences")
plt.savefig(INPUT_DIT + "/length_distribution.png")
plt.show()

seq_to_id = {seq: i for seq, i in meta_info_all[['sequence', 'precursor_index']].values}
id_to_seq = {v: k for v, k in meta_info_all[['precursor_index', 'sequence']].values}

shared_number = 7       # actually the first and last 6 amino acids
prefix_dict = defaultdict(list)
suffix_dict = defaultdict(list)
identical_dict = defaultdict(list)
cout = 0
for i in tqdm(meta_info_all[['precursor_index', 'sequence']].values):
    precursor_index = i[0]
    sequence = i[1]
    prefix = sequence[:shared_number]
    suffix = sequence[-shared_number:]
    prefix_dict[prefix].append(precursor_index)
    suffix_dict[suffix].append(precursor_index)
    identical_dict[sequence].append(precursor_index)
    cout += 1

print(f"identical_dict: {len(identical_dict)} unique identical sequences, {len([v for v in identical_dict.values() if len(v) > 1])} identical sequences with more than 1 precursor {sum([len(v) for v in identical_dict.values() if len(v) > 1])} precursors in total")
print(f"prefix_dict: {len(prefix_dict)} unique prefix, {len([v for v in prefix_dict.values() if len(v) > 1])} prefixes with more than 1 precursor {sum([len(v) for v in prefix_dict.values() if len(v) > 1])} precursors in total")
print(f"suffix_dict : {len(suffix_dict)} unique suffix, {len([v for v in suffix_dict.values() if len(v) > 1])} suffixes with more than 1 precursor {sum([len(v) for v in suffix_dict.values() if len(v) > 1])} precursors in total")

# create a graph, each node is a precursor_index, and each edge is a connection between two precursor_index if they have the same prefix or suffix
graph = defaultdict(set)
for prefix, precursor_indices in prefix_dict.items():
    for i in range(len(precursor_indices)):
        if len(precursor_indices) == 1:
            graph[precursor_indices[i]] = set()
        else:
            for j in range(i + 1, len(precursor_indices)):
                graph[precursor_indices[i]].add(precursor_indices[j])
                graph[precursor_indices[j]].add(precursor_indices[i])
        
for suffix, precursor_indices in suffix_dict.items():
    for i in range(len(precursor_indices)):
        if len(precursor_indices) == 1:
            graph[precursor_indices[i]] = set()
        else:
            for j in range(i + 1, len(precursor_indices)):
                graph[precursor_indices[i]].add(precursor_indices[j])
                graph[precursor_indices[j]].add(precursor_indices[i])
for identical, precursor_indices in identical_dict.items():
    for i in range(len(precursor_indices)):
        if len(precursor_indices) == 1:
            graph[precursor_indices[i]] = set()
        else:
            for j in range(i + 1, len(precursor_indices)):
                graph[precursor_indices[i]].add(precursor_indices[j])
                graph[precursor_indices[j]].add(precursor_indices[i])
# plot the distribution of the number of edges per node
num_edges_per_node = [len(graph[i]) for i in graph]
counter = collections.Counter(num_edges_per_node)
import matplotlib.pyplot as plt
plt.bar(counter.keys(), counter.values())
plt.xlabel("Number of edges per node")
plt.ylabel("Number of nodes")
plt.title("Distribution of number of edges per node")
plt.savefig(INPUT_DIT + "/num_edges_per_node.png")
plt.show()

# get a list of isolated subgraphs
def get_subgraph(graph, start_node):
    visited = set()
    stack = [start_node]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(graph[node])
    return visited

subgraphs = []
visited_nodes = set()
for node in graph:
    if node not in visited_nodes:
        subgraph = get_subgraph(graph, node)
        subgraphs.append(subgraph)
        visited_nodes.update(subgraph)
print("num_subgraphs:", len(subgraphs))

num_precursor_per_subgraph = [len(subgraph) for subgraph in subgraphs]
counter = collections.Counter(num_precursor_per_subgraph)
# save the counter as a tsv file
with open(os.path.join(INPUT_DIT, "num_precursor_per_subgraph.tsv"), "w") as f:
    f.write("num_precursor_per_subgraph\tcount\n")
    for key, value in counter.items():
        f.write(f"{key}\t{value}\n")
plt.figure(figsize=(5, 5))
plt.bar(counter.keys(), counter.values())
plt.xlabel("Number of precursors per subgraph")
plt.ylabel("Number of subgraphs")
plt.title("Distribution of number of precursors per subgraph")
plt.savefig(INPUT_DIT + "/num_precursor_per_subgraph.png")
plt.show()

# get the largest subgraph
subgraph_max_idx = np.argmax(num_precursor_per_subgraph)
largest_subgraph = list(subgraphs[subgraph_max_idx])
seq_largest_subgraph = [id_to_seq[i] for i in largest_subgraph]

# get the list of subgraphs of size 2
subgraph_size_2 = [subgraph for subgraph in subgraphs if len(subgraph) == 2]
# case 1: 39313, 39314, two identical sequences but different charge states
c1, c2 = 39313, 39314
seq1, seq2 = id_to_seq[c1], id_to_seq[c2]
c1_data = data_matrix[c1, :, 1]
c2_data = data_matrix[c2, :, 1]
mask_c1 = get_ion_mask(len(seq1), meta_info_all[meta_info_all.precursor_index == c1]["charge"].values[0], 40)
c1_data = c1_data[mask_c1]
c2_data = c2_data[mask_c1]
print("l1_distance:", compute_l1_distance(c1_data, c2_data) / sum(mask_c1))
print("cosine_distance:", compute_cosine(c1_data, c2_data))

# case 2: only the prefix of length 6 is the same
subgraph_size_6 = [list(subgraph) for subgraph in subgraphs if len(subgraph) == 6]
subgraph_seq = [[id_to_seq[i] for i in subgraph] for subgraph in subgraph_size_6]
c1, c2 = 9195, 9197
seq1, seq2 = id_to_seq[c1], id_to_seq[c2]
c1_data = data_matrix[c1, :, 1]
c2_data = data_matrix[c2, :, 1]
charge1 = meta_info_all[meta_info_all.precursor_index == c1]["charge"].values[0]
charge2 = meta_info_all[meta_info_all.precursor_index == c2]["charge"].values[0]
mask_c1 = get_ion_mask(len(seq1), charge1, 40)
mask_c2 = get_ion_mask(len(seq2), charge2, 40)
if sum(mask_c1) > sum(mask_c2):
    mask_c1 = mask_c2
c1_data = c1_data[mask_c1]
c2_data = c2_data[mask_c1]
print(c1, seq1, charge1, c2, seq2, charge2)
print("l1_distance:", compute_l1_distance(c1_data, c2_data) / sum(mask_c1))
print("cosine_distance:", compute_cosine(c1_data, c2_data))

# case 2: only the suffix of length 6 is the same
c1, c2 = 143303, 180753
seq1, seq2 = id_to_seq[c1], id_to_seq[c2]
c1_data = data_matrix[c1, :, 1]
c2_data = data_matrix[c2, :, 1]
charge1 = meta_info_all[meta_info_all.precursor_index == c1]["charge"].values[0]
charge2 = meta_info_all[meta_info_all.precursor_index == c2]["charge"].values[0]
mask_c1 = get_ion_mask(len(seq1), charge1, 40)
mask_c2 = get_ion_mask(len(seq2), charge2, 40)
if sum(mask_c1) > sum(mask_c2):
    mask_c1 = mask_c2
c1_data = c1_data[mask_c1]
c2_data = c2_data[mask_c1]
print(c1, seq1, charge1, c2, seq2, charge2)
print("l1_distance:", compute_l1_distance(c1_data, c2_data) / sum(mask_c1))
print("cosine_distance:", compute_cosine(c1_data, c2_data))

# sequence_list = meta_info_all["sequence"].tolist()
# num_seq = len(sequence_list)
# # prefixs and suffixes
# if not os.path.exists(prefix_suffix_low_dist_set_dict_path):
#     generate_prefix_suffix_low_dist_set_dict(sequence_list, max_low_dist, prefix_suffix_low_dist_set_dict_path)
#     print("prefix_suffix_low_dist_set_dict.pkl generated.")
# with open(prefix_suffix_low_dist_set_dict_path, "rb") as f:
#     prefix_suffix_low_dist_set_dict = pickle.load(f)

# ======= train_test split =======
sorted_subgraph = sorted(subgraphs, key=lambda x: len(x), reverse=True)
set_5 = {f'set_{i}': set() for i in range(5)}
set_5_num_isolated_subgraph = {f'set_{i}': 0 for i in range(5)}
while len(sorted_subgraph) > 0:
    subgraph = sorted_subgraph.pop(0)
    # get the smallest set of the 5 sets
    min_set_key = min(set_5, key=lambda x: len(set_5[x]))
    # add the subgraph to the set
    set_5[min_set_key].update(subgraph)
    set_5_num_isolated_subgraph[min_set_key] += 1
for key in set_5:
    print(f"{key}: {len(set_5[key])} precursors, num_isolated_subgraph: {set_5_num_isolated_subgraph[key]}")
with open(os.path.join(INPUT_DIT, "set_5_num_isolated_subgraph.tsv"), "w") as f:
    f.write("set_num\tprecursor_num\tnum_isolated_subgraph\n")
    for key in set_5_num_isolated_subgraph:
        f.write(f"{key}\t{len(set_5[key])}\t{set_5_num_isolated_subgraph[key]}\n")

for key in set_5:
    precursor_indices = np.array(list(set_5[key])).astype(int)
    # save the precursor_indices to a file
    np.save(os.path.join(INPUT_DIT, f"{key}.npy"), precursor_indices)

    # split the precursor_indices into train and test sets
    test_indices = precursor_indices
    train_set = set()
    for key2 in set_5:
        if key2 != key:
            train_set.update(set_5[key2])
    train_set = np.array(list(train_set))
    train_test = {'train_indices': train_set, 'test_indices': test_indices}
    np.save(os.path.join(INPUT_DIT, f"train_test_split_{key}.npy"), train_test)


# random 1000 pairs of precursor indices for identical sequences, prefixes and suffixes
n = 1000
identical_pairs, prefix_pairs, suffix_pairs = [], [], []
for identical in identical_dict:
    precursor_indices = identical_dict[identical]
    if len(precursor_indices) > 1:
        for i in range(len(precursor_indices)):
            for j in range(i + 1, len(precursor_indices)):
                identical_pairs.append((precursor_indices[i], precursor_indices[j]))
                if len(identical_pairs) >= n:
                    break
for prefix in prefix_dict:
    precursor_indices = prefix_dict[prefix]
    if len(precursor_indices) > 1:
        for i in range(len(precursor_indices)):
            for j in range(i + 1, len(precursor_indices)):
                prefix_pairs.append((precursor_indices[i], precursor_indices[j]))
                if len(prefix_pairs) >= n:
                    break
for suffix in suffix_dict:
    precursor_indices = suffix_dict[suffix]
    if len(precursor_indices) > 1:
        for i in range(len(precursor_indices)):
            for j in range(i + 1, len(precursor_indices)):
                suffix_pairs.append((precursor_indices[i], precursor_indices[j]))
                if len(suffix_pairs) >= n:
                    break
different_isolated_subgraph_pairs = []
for i in range(n):
    i1, i2 = np.random.choice(len(subgraphs), 2, replace=False)
    different_isolated_subgraph_pairs.append((list(subgraphs[i1])[0], list(subgraphs[i2])[0]))
ramdom_paris = []
for i in range(n):
    i1, i2 = np.random.choice(len(meta_info_all), 2, replace=False)
    ramdom_paris.append((i1, i2))

precursor_info_path = '/home/flower/Data/ms_workspace/fragment_prediction/mt_data/dataset_May5/precursor_info.tsv'
matrix_path = '/home/flower/Data/ms_workspace/fragment_prediction/mt_data/dataset_May5/matrix_mz_prob_mean_var.npy'
output_dir = '/home/flower/Data/ms_workspace/fragment_prediction/dataset_statistics'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

precursor_df = pd.read_csv(precursor_info_path, sep='\t')
matrix = np.load(matrix_path, allow_pickle=True)
probabilities = matrix[:, :, 1].copy()
peak_mask = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=bool)
for i in range(matrix.shape[0]):
    seq_len = len(precursor_df['sequence'][i])
    charge = precursor_df['charge'][i]
    mask = get_ion_mask(seq_len, charge, 40)
    peak_mask[i, :] = mask
assert matrix[:, :, 1][~peak_mask].max() < 1e-8
probabilities[~peak_mask] = 0

identical_pair_sa, prefix_pair_sa, suffix_pair_sa, different_isolated_subgraph_pair_sa, random_pari_sa = [], [], [], [], []
for i in range(n):
    i1, i2 = identical_pairs[i]
    inter_mask = peak_mask[i1] & peak_mask[i2]
    cosine = compute_cosine(probabilities[i1][inter_mask], probabilities[i2][inter_mask])
    identical_pair_sa.append(get_masked_spectral_distance(cosine))
    # identical_pair_sa.append(cosine)
    i1, i2 = prefix_pairs[i]
    inter_mask = peak_mask[i1] & peak_mask[i2]
    cosine = compute_cosine(probabilities[i1][inter_mask], probabilities[i2][inter_mask])
    prefix_pair_sa.append(get_masked_spectral_distance(cosine))
    # prefix_pair_sa.append(cosine)
    i1, i2 = suffix_pairs[i]
    inter_mask = peak_mask[i1] & peak_mask[i2]
    cosine = compute_cosine(probabilities[i1][inter_mask], probabilities[i2][inter_mask])
    suffix_pair_sa.append(get_masked_spectral_distance(cosine))
    # suffix_pair_sa.append(cosine)
    i1, i2 = different_isolated_subgraph_pairs[i]
    inter_mask = peak_mask[i1] & peak_mask[i2]
    cosine = compute_cosine(probabilities[i1][inter_mask], probabilities[i2][inter_mask])
    different_isolated_subgraph_pair_sa.append(get_masked_spectral_distance(cosine))
    # different_isolated_subgraph_pair_sa.append(cosine)
    i1, i2 = ramdom_paris[i]
    inter_mask = peak_mask[i1] & peak_mask[i2]
    cosine = compute_cosine(probabilities[i1][inter_mask], probabilities[i2][inter_mask])
    random_pari_sa.append(get_masked_spectral_distance(cosine))
    # random_pari_sa.append(cosine)

data = [identical_pair_sa, prefix_pair_sa, suffix_pair_sa, different_isolated_subgraph_pair_sa]
median_list = [np.median(i) for i in data]
# box plot
plt.figure(figsize=(4, 4))
plt.boxplot(data, labels=['Identical', 'SharePrefix', 'ShareSuffix', 'NoConnection'])
# plt.violinplot(data)
for i in range(len(median_list)):
    if i == 0:
        plt.plot([i + 0.8, i + 1.2], [median_list[i], median_list[i]], color='orange', linewidth=2, label='Median')
    else:
        plt.plot([i + 0.8, i + 1.2], [median_list[i], median_list[i]], color='orange', linewidth=2)
    plt.text(i + 1, median_list[i], f'{median_list[i]:.2f}', ha='center', va='bottom', fontsize=10)
# the color of the boxplot
plt.xlabel('Type of Connection')
plt.ylabel('Normalized Spectral Angle (SA)')
# plt.title('Masked spectral distance distribution')
plt.xticks(rotation=15)
plt.ylim(0.0, 1.0)
plt.savefig(os.path.join(output_dir, 'masked_spectral_distance_distribution.png'))
# plt.legend(loc='upper right', bbox_to_anchor=(1, 1.13), ncol=2, fontsize=10, frameon=True)
plt.tight_layout()
plt.show()


# # num of unique sequences
# num_unique_sequence = meta_info_all["sequence"].unique().shape[0]
# print("num_unique_sequence:", num_unique_sequence)

# # num of precursor whose minimum distance to another precursor is less than max_low_dist, as well as the prefix and suffix
# for key in prefix_suffix_low_dist_set_dict:
#     num_prefix_suffix_low_dist_precursor = len(prefix_suffix_low_dist_set_dict[key])
#     print(f"num_prefix_suffix_low_dist_precursor({key}):", num_prefix_suffix_low_dist_precursor)

# construct a graph from the precursor_info.tsv
# node: precursor_index
# edge: two precursor_index are connected if they have the same precursor_id
