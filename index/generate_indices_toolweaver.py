import collections
import json
import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE

import os

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

dataset = "ToolBench"
# ckpt_path = "/data2/fangbowen/LC-Rec/Nov-24-2024_00-45-06/best_collision_model.pth"
# ckpt_path = "/data2/fangbowen/LC-Rec/Dec-19-2024_23-02-58/best_collision_model.pth"
# ckpt_path = "/data2/fangbowen/LC-Rec/Dec-19-2024_23-02-58/best_loss_model.pth"
# ckpt_path = "/data2/fangbowen/LC-Rec/Feb-11-2025_12-32-28/best_collision_model.pth"
# ckpt_path = "/data2/fangbowen/LC-Rec/Feb-13-2025_12-03-02/best_collision_model.pth"
# ckpt_path = "/data2/fangbowen/Feb-19-2025_21-13-04/best_collision_model.pth"
# ckpt_path = "/data2/fangbowen/LC-Rec/Mar-05-2025_23-25-33/best_loss_model.pth"
# ckpt_path = "/vepfs-mlp/project_battery/public/fangbowen/LC-Rec/Jul-10-2025_16-12-33-graph_lambda-0.01/best_collision_model.pth"
ckpt_path = "./checkpoints/best_collision_model.pth"



output_dir = f"./output/"
output_file = f"{dataset}.index.1024.1024.random-embeddings.json"
output_file = os.path.join(output_dir,output_file)
device = torch.device("cuda:0")

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
args = ckpt["args"]
state_dict = ckpt["state_dict"]


data = EmbDataset(args.data_path)

model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  )

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print(model)

data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)

prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]
collision_prefix = "<f_{}>"  # Additional prefix for colliding items

all_indices = []
all_indices_str = []
collision_count = collections.defaultdict(int)  # Track collision counts for each item

for d in tqdm(data_loader):
    d = d.to(device)
    indices = model.get_indices(d,use_sk=False)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):
            code.append(prefix[i].format(int(ind)))
        all_indices.append(code)
        all_indices_str.append(str(code))
    # break

all_indices = np.array(all_indices)
all_indices_str = np.array(all_indices_str)

for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon=0.0
# model.rq.vq_layers[-1].sk_epsilon = 0.005
if model.rq.vq_layers[-1].sk_epsilon == 0.0:
    model.rq.vq_layers[-1].sk_epsilon = 0.003

tt = 0
#There are often duplicate items in the dataset, and we no longer differentiate them
while True:
    if tt >= 20 or check_collision(all_indices_str):
        break

    collision_item_groups = get_collision_item(all_indices_str)
    print(collision_item_groups)
    print(len(collision_item_groups))
    
    # First try to resolve conflicts using use_sk=True
    for collision_items in collision_item_groups:
        d = data[collision_items].to(device)
        indices = model.get_indices(d, use_sk=True)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for item, index in zip(collision_items, indices):
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))
            all_indices[item] = code
            all_indices_str[item] = str(code)
    
    # If there are still conflicts after use_sk, add collision tokens
    if not check_collision(all_indices_str):
        break
        
    collision_item_groups = get_collision_item(all_indices_str)
    for collision_items in collision_item_groups:
        # Add collision token for each item in the group
        for item in collision_items:
            collision_count[item] += 1
            code = all_indices[item].copy()
            code.append(collision_prefix.format(collision_count[item]))
            all_indices[item] = code
            all_indices_str[item] = str(code)
    
    tt += 1


print("All indices number: ",len(all_indices))
print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str.tolist()))
print("Collision Rate",(tot_item-tot_indice)/tot_item)

all_indices_dict = {}
for item, indices in enumerate(all_indices.tolist()):
    all_indices_dict[item] = list(indices)



with open(output_file, 'w') as fp:
    json.dump(all_indices_dict,fp)
