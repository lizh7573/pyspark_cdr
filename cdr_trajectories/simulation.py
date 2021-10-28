"""
Simulation
==========
"""

import numpy as np
import pyspark.sql.functions as F
import scipy.sparse as sparse
from pyspark.sql import Window
from numpy.linalg import matrix_power
from cdr_trajectories.TM import TM
from cdr_trajectories.udfs import prepare_for_sparse_plot
from cdr_trajectories.trajectories import time_inhomo_deterministic_trajectories,\
time_inhomo_probabilistic_trajectories

M = prepare_for_sparse_plot(TM(time_inhomo_deterministic_trajectories).make_tm(), 'updates')\
                       .toarray().tolist()

class Simulation:

    def __init__(self, df):
        self.df = df

    def vectorize(self):
        self.df = self.df.withColumn('v_col', F.col('states').__getitem__('neighbors'))\
                         .withColumn('v_val', F.col('states').__getitem__('props'))\
                         .withColumn('array_size', F.size(F.col('v_col')))\
                         .withColumn('v_row', F.expr('array_repeat(0, array_size)'))\
                         .select('voronoi_id', 'user_id', 'timestamp', 'v_row', 'v_col', 'v_val')
        return self.df

    def set_helpCol(self):
        window = Window.partitionBy(['user_id']).orderBy('timestamp')
        self.df = self.df.withColumn('i', F.row_number().over(window))
        return self.df

    def update(self):
        self.vectorize()
        self.set_helpCol()
        return self.df

def vectorize(x):

    voronoi_id = x.voronoi_id
    user_id = x.user_id
    timestamp = x.timestamp
    v_row = x.v_row
    v_col = x.v_col
    v_val = x.v_val
    i = x.i
    vector = sparse.coo_matrix((v_val, (v_row, v_col)), shape=(1,114+1)).toarray().tolist()
    matrix = matrix_power(M, i).tolist()
    update = np.dot(vector, matrix).tolist()

    return (voronoi_id, user_id, timestamp, v_row, v_col, v_val, i, vector, matrix, update)   

rdd2 = Simulation(time_inhomo_deterministic_trajectories).update()\
       .rdd.map(lambda x: vectorize(x))\
       .toDF(['voronoi_id', 'user_id', 'timestamp', 'v_row', 'v_col', 'v_val', 'i', 'vector', 'matrix', 'update'])

# print(type(rdd2))
# rdd2.toPandas().to_csv('result.csv')
# print(  np.shape(rdd2.toPandas()['update'][0])   )
# print(  np.shape(rdd2.toPandas()['matrix'][0])   )
# print(rdd2.toPandas()['update'][0])
# print(rdd2.toPandas()['update'][10])
print(np.sum(rdd2.toPandas()['update'][0]))
print(np.sum(rdd2.toPandas()['update'][1]))
print(np.sum(rdd2.toPandas()['update'][2]))
print(np.sum(rdd2.toPandas()['update'][3]))
print(np.sum(rdd2.toPandas()['update'][4]))
print(np.sum(rdd2.toPandas()['update'][5]))
print(np.sum(rdd2.toPandas()['update'][6]))
print(np.sum(rdd2.toPandas()['update'][7]))
