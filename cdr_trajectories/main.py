"""
Main module
===========
"""

import pyspark.sql.functions as F
from cdr_trajectories.mpn import mpn_data
from cdr_trajectories.voronoi import voronoi_data
from cdr_trajectories.ring import ring_data
from cdr_trajectories.TM import TM
from cdr_trajectories.time_inhomo import time_inhomo
from cdr_trajectories.udfs import prepare_for_plot, plot_dense, plot_sparse

trajectories = mpn_data.join(voronoi_data, ['avg_X', 'avg_Y'], how = 'inner')\
                       .orderBy(['user_id', 'timestamp'])

cdr_trajectories = trajectories.join(ring_data, ['voronoi_id'], how = 'inner')\
                               .orderBy(['user_id', 'timestamp'])\
                               .drop('avg_X', 'avg_Y', 'neighbors', 'props')



tm = TM(cdr_trajectories).make_tm()
print("Transition Matrix: Whole Time")
plot_dense(prepare_for_plot(tm, 'TM_updates'), 'TM.png', 'Transition Matrix')

#Paremeters are subjected to change
tm_time = TM(time_inhomo(cdr_trajectories, 6, 8).make_tm_time()).make_tm()
print("Transition Matrix: From 6am to 8am")
plot_dense(prepare_for_plot(tm_time, 'TM_updates'), 'TM (specific).png', 'Transition Matrix (6am to 8am)')



