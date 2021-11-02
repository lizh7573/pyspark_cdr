"""
Simulation
==========
"""

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from pyspark.sql import Window
from numpy.linalg import matrix_power
from cdr_trajectories.TM import TM
from cdr_trajectories.udfs import prepare_for_plot
from cdr_trajectories.trajectories import time_inhomo_deterministic_trajectories,\
time_inhomo_probabilistic_trajectories


M = prepare_for_plot(TM(time_inhomo_probabilistic_trajectories).make_tm(), 'updates')\
                       .toarray().tolist()


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
        matrix = matrix_power(M, i).tolist()
        vector = (np.dot(init_vector, matrix).tolist())[0]

        return (user_id, timestamp, vector)   


rdd2 = Vectorization(time_inhomo_probabilistic_trajectories).set_helpCols()\
       .rdd.map(lambda x: Vectorization.vectorize(x))\
       .toDF(['user_id', 'timestamp', 'vector'])\
       .withColumn('time', F.date_format('timestamp', 'HH:mm:ss'))\
       

# rdd2.toPandas().to_csv('result.csv')
# print( np.sum(rdd2.toPandas()['vector'][0]) )
# print( np.sum(rdd2.toPandas()['vector'][1]) )
# print( np.sum(rdd2.toPandas()['vector'][2]) )
# print( np.sum(rdd2.toPandas()['vector'][3]) )
# print( np.sum(rdd2.toPandas()['vector'][4]) )
# print( np.sum(rdd2.toPandas()['vector'][5]) )


rdd3 = rdd2.select(['time', 'vector']).groupBy('time')\
           .agg(F.array(*[F.avg(F.col('vector')[i]) for i in range(114+1)]).alias('vector'))\
           .orderBy('time')
           
           
# rdd3.toPandas().to_csv('result1.csv')

dfDistrHist = pd.DataFrame(rdd3.toPandas()['vector'])
# print(type(dfDistrHist))
dfDistrHist.plot()
plt.show()