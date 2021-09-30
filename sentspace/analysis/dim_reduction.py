from pathlib import Path
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import random
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import pickle
import os
from importlib import reload
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
import datetime
import sys
import copy
from tqdm import tqdm
import getpass
from sklearn.manifold import TSNE, MDS, Isomap
from os.path import join

corpora_identifier = 'switch'
source_model = 'gpt2'
sentence_embedding = 'avg-tok'
source_layer = 11

# dim reduction params
n_neighbors = 10
n_components = 3

user = getpass.getuser()
if user == 'gt':
	CACHEDIR = '/Users/gt/Documents/GitHub/control-neural/control_neural/model-actv-control/'
	CACHEDIR = '/Users/gt/om2/model-actv'
	DATADIR = (Path(os.getcwd()) / '..' / '..' / '..' / 'data').resolve()
else:
	CACHEDIR = f'/om2/user/{user}/model-actv/'
	DATADIR = f'/om2/user/{user}/corpora/'

CACHEFOLDER = os.path.join(CACHEDIR, source_model, sentence_embedding, )

actv = pd.read_pickle(join(CACHEFOLDER, f'{corpora_identifier}_activations.pkl'))

Isomap(n_neighbors, n_components)
MDS(n_components, max_iter=100, n_init=1)
TSNE(n_components=n_components, init='pca',
                                 random_state=0)



# Do 2d and 3d plots



# RUN PCA
actv_all_sents = bo['activations'].to_numpy()
actv_all_sents = np.concatenate(actv_all_sents)
pca = PCA(n_components=3)  # X.shape[0]
PC = pca.fit_transform(actv_all_sents)  # transform same input data into the new basis. Sklearn mean centers.
explained_var = pca.explained_variance_ratio_
plt.plot(np.cumsum(explained_var))
plt.show()
total_var = explained_var.sum() * 100
fig = go.Figure(auto_open=False,
     filename='plot_result.html')

# PCA
PC_merge = pca.fit_transform(merge_actv)  # transform same input data into the new basis. Sklearn mean centers.
explained_var = pca.explained_variance_ratio_

labels = {
	str(i): f"PC {i + 1} ({var:.1f}%)"
	for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}
fig = px.scatter_matrix(
	PC_merge,
	labels=labels,
	opacity=0.4,
	dimensions=range(np.shape(PC_merge)[1]),
	color=merge_info['score'],
	symbol=merge_info['type'],
	symbol_sequence=[2, 0],
	hover_name=merge_info['sentence'])
fig.update_traces(diagonal_visible=False)
fig.update_layout(coloraxis_colorbar=dict(
	title="Score"))
fig.show()

# drop activations from memory
del actv_all_sents

# simple 3d scatter:
fig = px.scatter_3d(
	PC, x=0, y=1, z=2, color=all_sents['score'],
	title=f'Total Explained Variance: {total_var:.2f}%',
	labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
)
fig.show()