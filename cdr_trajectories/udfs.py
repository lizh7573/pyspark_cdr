"""
User Defined Functions
======================
"""


import os
from posixpath import dirname
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt



def matrix_updates(states1, states2):
    update = [[float(el1[0]), float(el2[0]), el1[1]*el2[1]]
             for el1 in states1 for el2 in states2]
    return update


def prepare_for_sparse_plot(data, type_):

    pd_df = data.toPandas()

    data = np.array( pd_df[type_] )
    rows = np.array( pd_df['y'].astype('int') )
    cols = np.array( pd_df['x'].astype('int') )

    A = sparse.coo_matrix((data, (rows, cols)), shape = (114, 114))

    return A

def prepare_for_dense_plot(data, type_):

    pd_df = data.toPandas()

    data = np.array( pd_df[type_] )
    rows = np.array( pd_df['y'].astype('int') )
    cols = np.array( pd_df['x'].astype('int') )

    A = sparse.coo_matrix((data, (rows, cols)))

    return A


def plot_sparse(matrix, fname, title, dirname):
    plt.figure(figsize = (20, 20))
    plt.spy(matrix, markersize = 10, alpha = 0.5)
    plt.grid()
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xlabel("polygon", fontsize = 30)
    plt.ylabel("polygon", fontsize = 30)
    plt.title(title, fontsize = 35)
    plt.savefig(os.path.join(dirname, fname))


def plot_dense(matrix, fname, title, dirname):
    plt.figure(figsize = (20, 20))
    plt.imshow(matrix.todense())
    plt.colorbar()
    plt.grid()
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xlabel("polygon", fontsize = 30)
    plt.ylabel("polygon", fontsize = 30)
    plt.title(title, fontsize = 35)
    plt.savefig(os.path.join(dirname, fname))



