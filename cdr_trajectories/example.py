"""
Main Module
===========
"""

import os
import numpy as np
import pandas as pd
from random import seed, random

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import IntegerType

from cdr_trajectories.mpn import MPN
from cdr_trajectories.voronoi import Voronoi
from cdr_trajectories.ring import get_oneRingData, get_twoRingData, get_threeRingData
from cdr_trajectories.trajectories import DetermTraj, ProbTraj
from cdr_trajectories.time_inhomo import Time_inhomo
from cdr_trajectories.OD import OD
from cdr_trajectories.TM import TM
from cdr_trajectories.simulation import Vectorization, Simulation
from cdr_trajectories.udfs import prepare_for_plot, plot_sparse, plot_dense, vectorize, stationary,\
                                  vectorConverge, plot_trend, plot_result, simulate, plot_sim_result, rand_state


spark = SparkSession.builder\
    .enableHiveSupport()\
    .appName('cdr_trajectories')\
    .master('local[*]')\
    .getOrCreate()


mpn_file = 'data/mpn/*'
voronoi_file = 'data/voronoi/*'

firstRing_file = 'data/ring/voronoi20191213uppsla1st.txt'
secondRing_file = 'data/ring/voronoi20191213uppsla2d.txt'
thirdRing_file = 'data/ring/voronoi20191213uppsla3d.txt'



### 0 - PROCESSED DATASET:


## 0.1 - Mobile Phone Network:
mpn_data = MPN(spark, mpn_file).process()

## 0.2 - Vornnoi Polygon:
voronoi_data = Voronoi(spark, voronoi_file).process()

## 0.3 - Ring Distribution:
oneRing_data = get_oneRingData(spark, firstRing_file, secondRing_file, thirdRing_file)
twoRing_data = get_twoRingData(spark, firstRing_file, secondRing_file, thirdRing_file)
threeRing_data = get_threeRingData(spark, firstRing_file, secondRing_file, thirdRing_file)

## 0.4 - Deterministic Trajectories:
deterministic_trajectories = DetermTraj(mpn_data, voronoi_data).make_traj()

## 0.5 - Probabilistic Trajectories:
trajectories = DetermTraj(mpn_data, voronoi_data).join()

trajectories_oneRing = ProbTraj(trajectories, oneRing_data).make_traj()
trajectories_twoRing = ProbTraj(trajectories, twoRing_data).make_traj()
trajectories_threeRing = ProbTraj(trajectories, threeRing_data).make_traj()

probabilistic_trajectories = trajectories_oneRing



### 1 - ORIGIN-DESTINATION MATRICES:

od = OD(probabilistic_trajectories).make_od()
plot_dense(prepare_for_plot(od, 'updates'), 'OD.png', 'Origin-Destination Matrix', 'outputs/od')




### 2 - TRANSITION MATRICES:

## 2.1 - Deterministic Trajectories:
tm_0 = TM(deterministic_trajectories).make_tm()
plot_sparse(prepare_for_plot(tm_0, 'updates'), 'TM_0.png', 'Transition Matrix (Deterministic)', 'outputs/determTraj')


## 2.2 - Probabilistic Trajectories:
tm_1 = TM(probabilistic_trajectories).make_tm()
plot_sparse(prepare_for_plot(tm_1, 'updates'), 'TM_1.png', 'Transition Matrix (One Ring): 75% Confidence Level', 'outputs/probTraj')
plot_dense(prepare_for_plot(tm_1, 'updates'), 'TM_1_Dense.png', 'Transition Matrix (Probabilistic)', 'outputs/probTraj')

tm_2 = TM(trajectories_twoRing).make_tm()
plot_sparse(prepare_for_plot(tm_2, 'updates'), 'TM_2.png', 'Transition Matrix (Two Rings): 90% Confidence Level', 'outputs/probTraj')

tm_3 = TM(trajectories_threeRing).make_tm()
plot_sparse(prepare_for_plot(tm_3, 'updates'), 'TM_3.png', 'Transition Matrix (Three Rings): 95% Confidence Level', 'outputs/probTraj')










### 3 - TIME-INHOMOGENEOUS SIMULATION:

## 3.0 - Time-inhomogeneous Trajectories: Probabilistic
## Paremeters are subjected to change
time_inhomogeneous_prob_traj_0 = Time_inhomo(probabilistic_trajectories, 1, 5, 2, 3).make_ti_traj()
time_tm_0 = TM(time_inhomogeneous_prob_traj_0).make_tm()
plot_dense(prepare_for_plot(time_tm_0, 'updates'), 'TI_TM_0.png', 'Transition Matrix (2pm to 3pm)', 'outputs/time_inhomo')

time_inhomogeneous_prob_traj_1 = Time_inhomo(probabilistic_trajectories, 1, 5, 12, 18).make_ti_traj()
time_tm_1 = TM(time_inhomogeneous_prob_traj_1).make_tm()
plot_dense(prepare_for_plot(time_tm_1, 'updates'), 'TI_TM_1.png', 'Transition Matrix (1pm to 6pm)', 'outputs/time_inhomo')

time_inhomogeneous_prob_traj_2 = Time_inhomo(probabilistic_trajectories, 1, 5, 18, 20).make_ti_traj()
time_tm_2 = TM(time_inhomogeneous_prob_traj_2).make_tm()
plot_dense(prepare_for_plot(time_tm_2, 'updates'), 'TI_TM_2.png', 'Transition Matrix (6pm to 8pm)', 'outputs/time_inhomo')



## 3.1 - Stationary Distribution:

# Initial vector:
init_vector = Vectorization(time_inhomogeneous_prob_traj_1)\
              .process()\
              .rdd.map(lambda x: vectorize(x))\
              .toDF(['ml_SparseVector', 'np_vector'])
matrix_1 = prepare_for_plot(time_tm_1, 'updates').toarray()

init_vector_dev = stationary(500, init_vector, matrix_1)
init_vector_dev = init_vector_dev.loc[:, (init_vector_dev != 0).any(axis = 0)]

vector_1 = init_vector_dev.iloc[[0, -1]]

# Next vector:
next_vector = vectorConverge(spark, init_vector_dev)\
              .rdd.map(lambda x: vectorize(x))\
              .toDF(['ml_SparseVector', 'np_vector'])
matrix_2 = prepare_for_plot(time_tm_2, 'updates').toarray()

next_vector_dev = stationary(100, next_vector, matrix_2)
next_vector_dev = next_vector_dev.loc[:, (next_vector_dev != 0).any(axis = 0)]

vector_2 = next_vector_dev.iloc[-1:]

# Testing for convergence:
plot_trend(init_vector_dev, 'SD1_dev.png', 'Convergence Testing (Phase 1)', 'outputs/simulation')
plot_trend(next_vector_dev, 'SD2_dev.png', 'Convergence Testing (Phase 2)', 'outputs/simulation')

# Comparison: Beginning versus End
res = vector_1.append(vector_2).set_index([pd.Index([0,1,2])])
plot_result(res, 'SD.png', 'Mobility Trend', 'outputs/simulation')





## 3.2 - Simulate discrete markov chain:

# Get initial state for each user
# window = Window.partitionBy(['user_id']).orderBy(F.lit('A'))

# init_state = time_inhomogeneous_prob_traj_1\
#              .withColumn('i', F.row_number().over(window))\
#              .filter(F.col('i') == 1).select('user_id', 'voronoi_id')

# seed(0)
# sim_traj = spark.createDataFrame(simulate(init_state, matrix_1, 150, matrix_2, 50))

# sim_traj = sim_traj.withColumn('simulated_traj', F.explode(F.split(F.col('simulated_traj'), ',')))\
#                    .withColumn('simulated_traj', F.regexp_replace('simulated_traj', '\\[', ''))\
#                    .withColumn('simulated_traj', F.regexp_replace('simulated_traj', '\\]', ''))\
#                    .withColumn('simulated_traj', F.col('simulated_traj').cast(IntegerType()))\
#                    .withColumn('i', F.row_number().over(window))

# simulation = Simulation(sim_traj, oneRing_data).process()
# prepare_for_sim_plot = pd.DataFrame(np.vstack(simulation.toPandas()['vector']))
# plot_sim_result(prepare_for_sim_plot, 'Sim.png', 'Simulation Result', 'outputs/simulation')


# simulation_TM = Simulation(sim_traj, oneRing_data).reformulate_TM()
# sim_time_tm = TM(simulation_TM).make_sim_tm()
# plot_dense(prepare_for_plot(sim_time_tm, 'updates'), 'SIM_TI_TM.png', 'Simulated Transition Matrix (1pm to 7pm)', 'outputs/simulation')




### 4 - SCALIBILITY TEST IN DATABRICKS

# Create Random states
# Paremeters are subjected to change (randUser: number of users; randState: number of frequency)
# randUser = spark.range(1, 101)
# randState = rand_state(randUser, 24)

# Test Scalability with Transition Matrix
# test_TM = Simulation(randState, oneRing_data).reformulate_TM()
# test_tm = TM(test_TM).make_sim_tm()
# plot_dense(prepare_for_plot(test_tm, 'updates'), 'TEST_TM.png', 'Scalability Test - Simulated Transition Matrix', 'outputs/scalability')





