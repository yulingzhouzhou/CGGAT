from model.CGGAT import CGGAT
from jarvis.core.graphs import Graph
import numpy as np
import pandas as pd
from jarvis.db.figshare import data as jdata
from jarvis.core.atoms import Atoms
from data import get_train_val_loaders
import torch
import os

train, val, test, batch = get_train_val_loaders()
model = CGGAT()
path = "best_model.pt"
model.load_state_dict(torch.load(path, map_location='cpu')['model'])
# exit()
# total_params = sum(p.numel() for p in model.parameters())
# print(total_params)
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device).eval()
# exit()
out_list = []
target_list =[]
with torch.no_grad():
    for data in test:
        outputs = model([data[0].to(device), data[1].to(device)])
        target = data[2].to(device)
        outputs = outputs.to("cpu")
        target = target.to("cpu")
        out_list.append(outputs)
        target_list.append(target)
        print(outputs.shape)
out_list = np.array(out_list)
out_list = np.squeeze(out_list)
print(out_list.shape)
target_list = np.array(target_list)
print(target_list.shape)

import matplotlib.pyplot as plt
from sklearn import manifold
import matplotlib
matplotlib.use('TkAgg')
cmap = plt.get_cmap('hot')
z = manifold.TSNE(n_components=2, init='pca').fit_transform(out_list)
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(z[:, 0], z[:, 1], s=15, c=target_list, alpha=0.8, cmap=cmap)
cbar = plt.colorbar(scatter, ax=ax)
plt.rcParams['font.family'] = 'Time New Roman'
plt.rcParams['font.size'] = 18
# cbar.set_label('color')
plt.savefig("MBJ.eps", format='eps')
plt.show()
