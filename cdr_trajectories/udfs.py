"""
User Defined Functions
======================
"""


import os
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from random import seed, random
from pyspark.ml.linalg import Vectors




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



# def plot_sim_vector(vector, fname, title, dirname):

#     vector = vector.toPandas()
#     init_vector = vector['vector'][0]
#     vectorization = np.array([init_vector])

#     for x in range(1, len(vector.index)):
#         next_vectorization = np.array([vector['vector'][x]])
#         vectorization = np.append(vectorization, next_vectorization, axis = 0)
    
#     dfStationaryDist = pd.DataFrame(vectorization)   
#     dfStationaryDist.plot(legend = None)
#     plt.xlabel("iterated times", fontsize = 15)
#     plt.ylabel("probability", fontsize = 15)
#     plt.title(title, fontsize = 18)
#     plt.savefig(os.path.join(dirname, fname))


def plot_vector(vector, fname, title, dirname):
    
    dfStationaryDist = vector
    dfStationaryDist.plot(legend = None)

    plt.xlabel("iterated times", fontsize = 15)
    plt.ylabel("probability", fontsize = 15)
    # plt.yticks(np.arange(0, 0.6, 0.1))
    plt.title(title, fontsize = 18)
    plt.gcf().set_size_inches(16, 12)
    plt.savefig(os.path.join(dirname, fname))



def plot_vector_bar(vector, fname, title, dirname):

    last_vector = vector.tail(1)
    # last_vector.plot(legend = None)

    plt.figure(figsize = (16, 12))
    plt.bar(x = list(last_vector.columns), height = list(last_vector.iloc[0]))
    plt.xlabel("state", fontsize = 15)
    plt.ylabel("probability", fontsize = 15)
    plt.xticks(range(0, 114+1, 10))
    plt.title(title, fontsize = 18)
    plt.savefig(os.path.join(dirname, fname))


# def randomize(current_row):
#     seed(4)
#     r = np.random.uniform(0.0, 1.0)
#     CS = np.cumsum(current_row)
#     m = (np.where(CS < r))[0]
#     nextState = m[len(m)-1]+1
#     return nextState



# def simulate(data):

#     df = data.toPandas()
#     P = np.array( df['matrix'][0] )

#     for i in df.index:
#         currentState = df['voronoi_id'][i]
#         simulated_traj = [currentState.item()]

#         for x in range(1000):
#             currentRow = np.ma.masked_values((P[currentState]), 0.0)
#             nextState = randomize(currentRow)
#             simulated_traj = simulated_traj + [nextState.item()]
#             currentState = nextState
        
#         df.at[i, 'simulated_traj'] = str(simulated_traj)

#     return df


def sim_vectorize(x):

    user_id = x.user_id
    simulated_traj = x.simulated_traj
    col = x.col
    val = x.val
    ml_SparseVector = Vectors.sparse(114+1, col, val)
    sim_vector = ml_SparseVector.toArray().tolist()
    i = x.i

    return (user_id, simulated_traj, sim_vector, i)






