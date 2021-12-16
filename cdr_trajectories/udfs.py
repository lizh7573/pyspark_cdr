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
import pyspark.sql.functions as F




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
    plt.xlabel("Polygon", fontsize = 27)
    plt.ylabel("Polygon", fontsize = 27)
    plt.title(title, fontsize = 30)
    plt.savefig(os.path.join(dirname, fname))


def plot_dense(matrix, fname, title, dirname):
    plt.figure(figsize = (20, 20))
    plt.imshow(matrix.todense())
    plt.colorbar()
    plt.grid()
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.xlabel("Polygon", fontsize = 27)
    plt.ylabel("Polygon", fontsize = 27)
    plt.title(title, fontsize = 30)
    plt.savefig(os.path.join(dirname, fname))


def plot_trend(vector, fname, title, dirname):
    
    dfStationaryDist = vector
    dfStationaryDist.plot()

    plt.xlabel("Iterated times", fontsize = 18)
    plt.ylabel("Probability", fontsize = 18)
    plt.title(title, fontsize = 20)
    plt.gcf().set_size_inches(16, 12)
    plt.savefig(os.path.join(dirname, fname))



def plot_result(vector, fname, title, dirname):

    labels = list(vector.columns)
    initial = list(vector.iloc[0])
    middle = list(vector.iloc[1])
    end = list(vector.iloc[2])

    X = np.arange(len(vector.columns))
    width = 0.2

    plt.figure(figsize = (16, 12))
    plt.bar(X - 0.2, initial, width, color = 'deepskyblue', label = 'Initial')
    plt.bar(X, middle, width, color = 'gold', label = 'Middle')
    plt.bar(X + 0.2, end, width, color = 'grey', label = 'End')
    plt.xticks(X, labels)
    plt.xlabel("Polygon", fontsize = 18)
    plt.ylabel("Probability", fontsize = 18)
    plt.legend(['Initial', 'Middle', 'End'], fontsize = 18)
    
    plt.title(title, fontsize = 20)
    plt.savefig(os.path.join(dirname, fname))



def vectorize(x):

    col = x.col
    val = x.val
    ml_SparseVector = Vectors.sparse(114+1, col, val)
    np_vector = ml_SparseVector.toArray().tolist()

    return (ml_SparseVector, np_vector) 


def stationary(n, vector, matrix):

    current_sv = vector.first()['ml_SparseVector']
    current_v = vector.first()['np_vector']
    res = np.array([current_v])

    for j in range(n-1):

        next_v = (current_sv.dot(matrix)).tolist()
        res = np.append( res, np.array([next_v]), axis = 0 )
        d = {x: next_v[x] for x in np.nonzero(next_v)[0]}
        next_sv = Vectors.sparse(len(next_v), d)
        current_sv = next_sv

    stationary_vector = pd.DataFrame(res)

    return stationary_vector


def vectorConverge(spark, vector):

    last_vector = vector.iloc[-1:]
    drop_zeros = last_vector.loc[:, (last_vector != 0).any(axis = 0)]
    transpose = drop_zeros.melt()

    next_vector = spark.createDataFrame(transpose)\
                          .agg(F.collect_list('variable').alias('col'),
                               F.collect_list('value').alias('val'))

    return next_vector



def randomize(current_row):
    r = np.random.uniform(0.0, 1.0)
    cum = np.cumsum(current_row)
    m = (np.where(cum < r))[0]
    nextState = m[len(m)-1]+1
    return nextState


def simulate(vector, matrix1, m, matrix2, n):

    df = vector.toPandas()
    P1 = matrix1
    P2 = matrix2

    for i in df.index:
        currentState = df['voronoi_id'][i]
        simulated_traj = [currentState.item()]

        for x in range(m):
            currentRow = np.ma.masked_values((P1[currentState]), 0.0)
            nextState = randomize(currentRow)
            simulated_traj = simulated_traj + [nextState.item()]
            currentState = nextState
        
        df.at[i, 'simulated_traj'] = str(simulated_traj)

        for y in range(n):
            currentRow = np.ma.masked_values((P2[currentState]), 0.0)
            nextState = randomize(currentRow)
            simulated_traj = simulated_traj + [nextState.item()]
            currentState = nextState
            
        df.at[i, 'simulated_traj'] = str(simulated_traj)  

    return df


def sim_vectorize(x):

    user_id = x.user_id
    simulated_traj = x.simulated_traj
    col = x.col
    val = x.val
    ml_SparseVector = Vectors.sparse(114+1, col, val)
    sim_vector = ml_SparseVector.toArray().tolist()
    i = x.i

    return (user_id, simulated_traj, sim_vector, i)


def plot_sim_result(vector, fname, title, dirname):

    last_vector = vector.iloc[-1:]

    plt.figure(figsize = (16, 12))
    plt.bar(x = list(last_vector.columns), height = list(last_vector.iloc[0]))
    plt.xlabel("Polygon", fontsize = 18)
    plt.ylabel("Probability", fontsize = 18)
    plt.xticks(range(0, 114+1, 10))
    plt.title(title, fontsize = 20)
    plt.savefig(os.path.join(dirname, fname))






