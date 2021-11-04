"""
User Defined Functions
======================
"""


import os
import numpy as np
import pandas as pd
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

    M = sparse.coo_matrix((data, (rows, cols)), shape = (114+1, 114+1))

    return M


def plot_sparse(matrix, fname, title, dirname):
    plt.figure(figsize = (20, 20))
    plt.spy(matrix, markersize = 10, alpha = 0.5)
    plt.grid()
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xlabel("polygon", fontsize = 27)
    plt.ylabel("polygon", fontsize = 27)
    plt.title(title, fontsize = 30)
    plt.savefig(os.path.join(dirname, fname))


def plot_dense(matrix, fname, title, dirname):
    plt.figure(figsize = (20, 20))
    plt.imshow(matrix.todense())
    plt.colorbar()
    plt.grid()
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xlabel("polygon", fontsize = 27)
    plt.ylabel("polygon", fontsize = 27)
    plt.title(title, fontsize = 30)
    plt.savefig(os.path.join(dirname, fname))



def plot_vector(vector, fname, title, dirname):

    vector = vector.toPandas()
    init_vector = vector['vector'][0]
    vectorization = np.array([init_vector])

    for x in range(1, len(vector.index)):
        next_vectorization = np.array([vector['vector'][x]])
        vectorization = np.append(vectorization, next_vectorization, axis = 0)
        dfStationaryDist = pd.DataFrame(vectorization)
        dfStationaryDist.plot(legend = None)
    
    plt.xlabel("iterated times", fontsize = 15)
    plt.ylabel("probability", fontsize = 15)
    plt.title(title, fontsize = 18)
    plt.savefig(os.path.join(dirname, fname))






