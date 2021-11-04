"""
Simulation
==========
"""

"""
import os

from PIL.Image import ROTATE_90
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from pyspark.sql import Window
from numpy.linalg import matrix_power
from cdr_trajectories.TM import TM
from cdr_trajectories.udfs import prepare_for_plot
from cdr_trajectories.trajectories import time_inhomo_probabilistic_trajectories
from cdr_trajectories.udfs import plot_vector
from cdr_trajectories.constants import spark

Matrix = prepare_for_plot(TM(time_inhomo_probabilistic_trajectories).make_tm(), 'updates').toarray().tolist()

class Vectorization:

    def __init__(self, df):
        self.df = df

    def set_helpCols(self):
        window = Window.partitionBy(['user_id']).orderBy('timestamp')
        self.df = self.df.withColumn('v_col', F.first(F.col('states').__getitem__('neighbors')).over(window))\
                         .withColumn('v_val', F.first(F.col('states').__getitem__('props')).over(window))\
                         .withColumn('array_size', F.size(F.col('v_col')))\
                         .withColumn('v_row', F.expr('array_repeat(0, array_size)'))\
                         .withColumn('i', F.row_number().over(window))\
                         .select('voronoi_id', 'user_id', 'timestamp', 'v_row', 'v_col', 'v_val', 'i')
        return self.df

    @staticmethod
    def vectorize(x):

        voronoi_id = x.voronoi_id
        user_id = x.user_id
        timestamp = x.timestamp
        v_row = x.v_row
        v_col = x.v_col
        v_val = x.v_val
        i = x.i
        init_vector = sparse.coo_matrix((v_val, (v_row, v_col)), shape=(1,114+1)).toarray().tolist()
        matrix = matrix_power(Matrix, i).tolist()
        vector = (np.dot(init_vector, matrix).tolist())[0]

        return (user_id, timestamp, vector)   

vector_data = Vectorization(time_inhomo_probabilistic_trajectories).set_helpCols()\
        .rdd.map(lambda x: Vectorization.vectorize(x))\
        .toDF(['user_id', 'timestamp', 'vector'])
        

class Stationary:

    def __init__(self, df):
        self.df = df

    def process(self):
        self.df = self.df\
        .withColumn('time', F.date_format('timestamp', 'HH:mm:ss'))\
        .select(['time', 'vector']).groupBy('time')\
        .agg(F.array(*[F.avg(F.col('vector')[i]) for i in range(114+1)]).alias('vector'))\
        .orderBy('time')
        return self.df

stationaryVector = Stationary(vector_data).process()

stationaryVector = stationaryVector.toPandas()
# print(len(stationaryVector.index))
test = stationaryVector['vector'][len(stationaryVector.index)-1]
# stationaryVector = spark.createDataFrame()
test1 = np.array([test])
test2 = pd.DataFrame(test1)
# print(list(test2.iloc[0]))
# test2.to_csv('test2.csv')
test2.plot(legend = None)

plt.bar(x = list(test2.columns), height = list(test2.iloc[0]))
plt.xticks(range(0, 114+1, 10))
plt.savefig('test.png')
# plot_vector(stationaryVector, 'SD.png', 'Stationary Distribution', 'outputs/simulation')

def plot_vector_bar(vector, fname, title, dirname):

    vector = vector.toPandas()
    last_vector = vector['vector'][len(vector.index)-1]
    vectorization = np.array([last_vector])
    dfStationaryDist = pd.DataFrame(vectorization)
    dfStationaryDist.plot(legend = None)

    plt.bar(x = list(dfStationaryDist.columns), height = list(dfStationaryDist.iloc[0]))
    plt.xlabel("state", fontsize = 15)
    plt.ylabel("stationary probability", fontsize = 15)
    plt.xticks(range(0, 114+1, 10))
    plt.title(title, fontsize = 18)
    plt.savefig(os.path.join(dirname, fname))



# stationaryVector = stationaryVector.toPandas()  
# init_vector = stationaryVector['vector'][0]   
# stateHist = np.array([init_vector])

# for x in range(1, len(stationaryVector.index)):
#     state = np.array([stationaryVector['vector'][x]])
#     stateHist = np.append(stateHist, state, axis = 0)
#     dfDistrHist = pd.DataFrame(stateHist)
#     dfDistrHist.plot(legend = None)

# plt.xlabel("iterated times", fontsize = 15)
# plt.ylabel("probability", fontsize = 15)
# plt.title("Stationary Distribution", fontsize = 18)
# plt.savefig(os.path.join('outputs/', 'test.png'))


"""