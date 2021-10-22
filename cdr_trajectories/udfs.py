import os
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

def matrix_updates(states1, states2):
    update = [[float(el1[0]), float(el2[0]), el1[1]*el2[1]]
             for el1 in states1 for el2 in states2]
    return update


def prepare_for_plot(data, type_):

    pd_df = data.toPandas()

    data = np.array( pd_df[type_] )
    rows = np.array( pd_df['y'].astype('int') )
    cols = np.array( pd_df['x'].astype('int') )

    A = sparse.coo_matrix((data, (rows, cols)))

    return A


def plot_dense(matrix, fname, title):
    plt.figure(figsize = (20, 20))
    plt.imshow(matrix.todense())
    plt.colorbar()
    plt.grid()
    plt.xlabel("polygon", fontsize = 20)
    plt.ylabel("polygon", fontsize = 20)
    plt.title(title, fontsize = 30)
    plt.savefig(os.path.join('outputs', fname))


def plot_sparse(matrix, fname, title):
    plt.figure(figsize = (20, 20))
    plt.spy(matrix, markersize = 4, alpha = 0.25)
    plt.grid()
    plt.xlabel("polygon", fontsize = 20)
    plt.ylabel("polygon", fontsize = 20)
    plt.title(title, fontsize = 30)
    plt.savefig(os.path.join('outputs', fname))
