"""
Main module
===========
"""

import os
import pyspark.sql.functions as F
from pyspark.sql.functions import arrays_zip
from cdr_trajectories.mpn import mpn_data
from cdr_trajectories.voronoi import voronoi_data
from cdr_trajectories.ring import oneRing_data, twoRing_data, threeRing_data
from cdr_trajectories.TM import TM
from cdr_trajectories.OD import OD
from cdr_trajectories.time_inhomo import time_inhomo
from cdr_trajectories.udfs import prepare_for_sparse_plot, prepare_for_dense_plot, plot_sparse, plot_dense

trajectories = mpn_data.join(voronoi_data, ['avg_X', 'avg_Y'], how = 'inner')\
                       .orderBy(['user_id', 'timestamp'])

# Deterministic Trajectories
zeroRing_trajectories = trajectories.withColumn('neighbors', F.array('voronoi_id'))\
                                    .withColumn('props', F.array(F.lit(1.0)))\
                                    .withColumn('states', arrays_zip('neighbors', 'props'))\
                                    .orderBy(['user_id', 'timestamp'])\
                                    .drop('avg_X', 'avg_Y', 'neighbors', 'props')

deterministic_trajectories = zeroRing_trajectories
deterministic_trajectories.toPandas().to_csv(os.path.join('outputs/determTraj', 'determTraj.csv'))

tm_0 = TM(deterministic_trajectories).make_tm()
plot_sparse(prepare_for_sparse_plot(tm_0, 'updates'), 'TM_0.png', 
           'Transition Matrix (Deterministic)', 'outputs/determTraj')
tm_0.toPandas().to_csv(os.path.join('outputs/determTraj', 'tm_0.csv'))


# Probabilistic Trajectories
class Trajectories:

    def __init__(self, df, ring):
        self.df = df
        self.ring = ring

    def join(self):
        self.df = self.df.join(self.ring, ['voronoi_id'], how = 'inner')\
                         .orderBy(['user_id', 'timestamp'])\
                         .drop('avg_X', 'avg_Y', 'neighbors', 'props')
        return self.df

oneRing_trajectories = Trajectories(trajectories, oneRing_data).join()
twoRing_trajectories = Trajectories(trajectories, twoRing_data).join()
threeRing_trajectories = Trajectories(trajectories, threeRing_data).join()

probabilistic_trajectories = threeRing_trajectories

tm_1 = TM(oneRing_trajectories).make_tm()
plot_dense(prepare_for_dense_plot(tm_1, 'updates'), 'TM_1.png',
          'Transition Matrix (One Ring)', 'outputs/probTraj')
tm_2 = TM(twoRing_trajectories).make_tm()
plot_dense(prepare_for_dense_plot(tm_2, 'updates'), 'TM_2.png', 
          'Transition Matrix (Two Rings)', 'outputs/probTraj')
tm_3 = TM(probabilistic_trajectories).make_tm()
plot_dense(prepare_for_dense_plot(tm_3, 'updates'), 'TM_3.png', 
          'Transition Matrix (Three Rings)', 'outputs/probTraj')



# Time-inhomogeneous Trajectories
# Paremeters are subjected to change
time_tm_0 = TM(time_inhomo(deterministic_trajectories, 4, 4, 6, 8).make_tm_time()).make_tm()
plot_sparse(prepare_for_sparse_plot(time_tm_0, 'updates'), 'specific_TM_0.png', 
            'Transition Matrix (Deterministic) (Thursday: 6am to 8am)', 'outputs/time_inhomo')
time_inhomo_deterministic_trajectories = time_inhomo(deterministic_trajectories, 4, 4, 6, 8).make_tm_time()
time_inhomo_deterministic_trajectories.toPandas().to_csv(os.path.join('outputs/time_inhomo', 'time_DetermTraj.csv'))
time_tm_0.toPandas().to_csv(os.path.join('outputs/time_inhomo', 'time_tm_0.csv'))



# Origin-Destination Matrices
od = OD(probabilistic_trajectories).make_od()
plot_dense(prepare_for_dense_plot(od, 'updates'), 'OD.png',
          'Origin-Destination Matrix', 'outputs/od')





