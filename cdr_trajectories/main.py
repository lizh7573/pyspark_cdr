"""
Main module
===========
"""

import os
import numpy as np
import pyspark.sql.functions as F
import scipy.sparse as sparse
from numpy.linalg import matrix_power
from cdr_trajectories.TM import TM
from cdr_trajectories.OD import OD
from cdr_trajectories.udfs import prepare_for_plot, plot_sparse, plot_dense, plot_vector, plot_vector_bar
from cdr_trajectories.simulation import Vectorization, Stationary
from cdr_trajectories.trajectories import deterministic_trajectories, oneRing_trajectories,\
twoRing_trajectories, probabilistic_trajectories, time_inhomo_deterministic_trajectories,\
time_inhomo_probabilistic_trajectories



## DETERMINISTIC TRAJECTORIES

tm_0 = TM(deterministic_trajectories).make_tm()
plot_sparse(prepare_for_plot(tm_0, 'updates'), 'TM_0.png', 
           'Transition Matrix (Deterministic)', 'outputs/determTraj')
# tm_0.toPandas().to_csv(os.path.join('outputs/determTraj', 'tm_0.csv'))
# deterministic_trajectories.toPandas().to_csv(os.path.join('outputs/determTraj', 'determTraj.csv'))



## PROBABILISTIC TRAJECTORIES

tm_1 = TM(oneRing_trajectories).make_tm()
plot_dense(prepare_for_plot(tm_1, 'updates'), 'TM_1.png',
          'Transition Matrix (One Ring)', 'outputs/probTraj')

tm_2 = TM(twoRing_trajectories).make_tm()
plot_dense(prepare_for_plot(tm_2, 'updates'), 'TM_2.png', 
          'Transition Matrix (Two Rings)', 'outputs/probTraj')

tm_3 = TM(probabilistic_trajectories).make_tm()
plot_dense(prepare_for_plot(tm_3, 'updates'), 'TM_3.png', 
          'Transition Matrix (Three Rings)', 'outputs/probTraj')



## TIME-INHOMOGENEOUS TRAJECTORIES
# Paremeters are subjected to change

# Deterministic
time_tm_0 = TM(time_inhomo_deterministic_trajectories).make_tm()
plot_sparse(prepare_for_plot(time_tm_0, 'updates'), 'specific_TM_0.png', 
            'Time Inhomogeneous Transition Matrix (Deterministic)', 'outputs/time_inhomo')
# time_inhomo_deterministic_trajectories.toPandas().to_csv(os.path.join('outputs/time_inhomo', 'time_DetermTraj.csv'))
# time_tm_0.toPandas().to_csv(os.path.join('outputs/time_inhomo', 'time_tm_0.csv'))

# Probabilistic
time_tm_3 = TM(time_inhomo_probabilistic_trajectories).make_tm()
plot_dense(prepare_for_plot(time_tm_3, 'updates'), 'specific_TM_3.png', 
            'Time Inhomogeneous Transition Matrix (Probabilistic)', 'outputs/time_inhomo')
time_inhomo_probabilistic_trajectories.toPandas().to_csv(os.path.join('outputs/time_inhomo', 'time_ProbTraj.csv'))
time_tm_3.toPandas().to_csv(os.path.join('outputs/time_inhomo', 'time_tm_3.csv'))




## ORIGIN-DESTINATION MATRICES
od = OD(probabilistic_trajectories).make_od()
plot_dense(prepare_for_plot(od, 'updates'), 'OD.png',
          'Origin-Destination Matrix', 'outputs/od')



## SIMULATION 

# Stationary Distribution
Matrix = prepare_for_plot(TM(time_inhomo_probabilistic_trajectories).make_tm(), 'updates').toarray().tolist()

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
        .rdd.map(lambda x: vectorize(x))\
        .toDF(['user_id', 'timestamp', 'vector'])

stationaryVector = Stationary(vector_data).process()

stationaryVector.toPandas().to_csv(os.path.join('outputs/simulation', 'time_sd_3.csv'))
plot_vector(stationaryVector, 'SD_dev.png', 'Stationary Distribution', 'outputs/simulation')
plot_vector_bar(stationaryVector, 'SD.png', 'Stationary Distribution', 'outputs/simulation')







