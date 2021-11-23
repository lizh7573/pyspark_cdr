"""
Main Module
===========
"""

import os
import numpy as np
import pandas as pd
from random import seed, random

import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import IntegerType

from cdr_trajectories.constants import Spark
from cdr_trajectories.mpn import MPN
from cdr_trajectories.voronoi import Voronoi
from cdr_trajectories.ring import get_oneRingData, get_twoRingData, get_threeRingData
from cdr_trajectories.trajectories import DetermTraj, ProbTraj
from cdr_trajectories.time_inhomo import Time_inhomo
from cdr_trajectories.OD import OD
from cdr_trajectories.TM import TM
from cdr_trajectories.simulation import Vectorization, Simulation
from cdr_trajectories.udfs import prepare_for_plot, plot_sparse, plot_dense, plot_vector, plot_vector_bar




mpn_file = 'data/mpn/*'
voronoi_file = 'data/voronoi/*'

firstRing_file = 'data/ring/voronoi20191213uppsla1st.txt'
secondRing_file = 'data/ring/voronoi20191213uppsla2d.txt'
thirdRing_file = 'data/ring/voronoi20191213uppsla3d.txt'



### 0 - PROCESSED DATASET:
## 0.1 - Mobile Phone Network:
mpn_data = MPN(mpn_file).process()
## 0.2 - Vornnoi Polygon:
voronoi_data = Voronoi(voronoi_file).process()
## 0.3 - Ring Distribution:
oneRing_data = get_oneRingData(firstRing_file, secondRing_file, thirdRing_file)
twoRing_data = get_twoRingData(firstRing_file, secondRing_file, thirdRing_file)
threeRing_data = get_threeRingData(firstRing_file, secondRing_file, thirdRing_file)
## 0.4 - Deterministic Trajectories:
deterministic_trajectories = DetermTraj(mpn_data, voronoi_data).make_traj()
## 0.5 - Probabilistic Trajectories:
trajectories = DetermTraj(mpn_data, voronoi_data).join()
trajectories_oneRing = ProbTraj(trajectories, oneRing_data).make_traj()
trajectories_twoRing = ProbTraj(trajectories, twoRing_data).make_traj()
probabilistic_trajectories = ProbTraj(trajectories, threeRing_data).make_traj()
time_inhomogeneous_probabilistic_trajectories = Time_inhomo(probabilistic_trajectories, 1, 5, 12, 18).make_ti_traj()




### 1 - ORIGIN-DESTINATION MATRICES:

od = OD(probabilistic_trajectories).make_od()
plot_dense(prepare_for_plot(od, 'updates'), 'OD.png', 'Origin-Destination Matrix', 'outputs/od')





### 2 - TRANSITION MATRICES:

## 2.1 - Deterministic Trajectories:
tm_0 = TM(deterministic_trajectories).make_tm()
plot_sparse(prepare_for_plot(tm_0, 'updates'), 'TM_0.png', 'Transition Matrix (Deterministic)', 'outputs/determTraj')




## 2.2 - Probabilistic Trajectories:
tm_1 = TM(trajectories_oneRing).make_tm()
plot_sparse(prepare_for_plot(tm_1, 'updates'), 'TM_1.png', 'Transition Matrix (One Ring)', 'outputs/probTraj')

tm_2 = TM(trajectories_twoRing).make_tm()
plot_sparse(prepare_for_plot(tm_2, 'updates'), 'TM_2.png', 'Transition Matrix (Two Rings)', 'outputs/probTraj')

tm_3 = TM(probabilistic_trajectories).make_tm()
plot_sparse(prepare_for_plot(tm_3, 'updates'), 'TM_3.png', 'Transition Matrix (Three Rings)', 'outputs/probTraj')
plot_dense(prepare_for_plot(tm_3, 'updates'), 'TM_3_Dense.png', 'Transition Matrix (Probabilistic)', 'outputs/probTraj')



## 2.3 - Time-inhomogeneous Trajectories: Probabilistic
## Paremeters are subjected to change
time_tm_3 = TM(time_inhomogeneous_probabilistic_trajectories).make_tm()
plot_dense(prepare_for_plot(time_tm_3, 'updates'), 'TI_TM_3.png', 'Time Inhomogeneous Transition Matrix (Probabilistic)', 'outputs/time_inhomo')




### 3 - SIMULATION:

## 3.1 - Stationary Distribution:


def vectorize(x):

    n = x.n
    col = x.col
    val = x.val
    ml_SparseVector = Vectors.sparse(114+1, col, val)
    np_vector = ml_SparseVector.toArray().tolist()

    return (n, ml_SparseVector, np_vector) 


vector_data = Vectorization(time_inhomogeneous_probabilistic_trajectories)\
              .process()\
              .rdd.map(lambda x: vectorize(x))\
              .toDF(['n', 'ml_SparseVector', 'np_vector'])

matrix = prepare_for_plot(TM(time_inhomogeneous_probabilistic_trajectories).make_tm(), 'updates').toarray()


def stationary(vector):

    n = vector.first()['n']
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


plot_vector(stationary(vector_data), 'SD_dev.png', 'Stationary Distribution', 'outputs/simulation')
plot_vector_bar(stationary(vector_data), 'SD.png', 'Stationary Distribution', 'outputs/simulation')






## 3.2 - Simulate discrete markov chain:
w1 = Window.partitionBy(['user_id']).orderBy(F.lit('A'))
init_state = time_inhomogeneous_probabilistic_trajectories\
             .withColumn('i', F.row_number().over(w1))\
             .filter(F.col('i') == 1).select('user_id', 'voronoi_id')

def randomize(current_row):
    seed(4)
    r = np.random.uniform(0.0, 1.0)
    cum = np.cumsum(current_row)
    m = (np.where(cum < r))[0]
    nextState = m[len(m)-1]+1
    return nextState

def simulate(data):

    df = data.toPandas()
    P = matrix

    for i in df.index:
        currentState = df['voronoi_id'][i]
        simulated_traj = [currentState.item()]

        for x in range(1000):
            currentRow = np.ma.masked_values((P[currentState]), 0.0)
            nextState = randomize(currentRow)
            simulated_traj = simulated_traj + [nextState.item()]
            currentState = nextState
        
        df.at[i, 'simulated_traj'] = str(simulated_traj)

    return df

init_sim = Spark.createDataFrame(simulate(init_state))

w2 = Window.partitionBy(['user_id']).orderBy(F.lit('A'))
init_sim = init_sim.withColumn('simulated_traj', F.explode(F.split(F.col('simulated_traj'), ',')))\
         .withColumn('simulated_traj', F.regexp_replace('simulated_traj', '\\[', ''))\
         .withColumn('simulated_traj', F.regexp_replace('simulated_traj', '\\]', ''))\
         .withColumn('simulated_traj', F.col('simulated_traj').cast(IntegerType()))\
         .withColumn('i', F.row_number().over(w2))\
         .select('user_id', 'simulated_traj', 'i')


sim_traj = init_sim.join(F.broadcast(threeRing_data), init_sim.simulated_traj == threeRing_data.voronoi_id, how = 'inner')\
              .orderBy(['user_id', 'i']).select('user_id', 'simulated_traj', 'states', 'i')


simulation = Simulation(sim_traj).process()
prepare_for_sim_plot = pd.DataFrame(np.vstack(simulation.toPandas()['vector']))


plot_vector(prepare_for_sim_plot, 'Sim_dev.png', 'Simulation', 'outputs/simulation')
plot_vector_bar(prepare_for_sim_plot, 'Sim.png', 'Simulation Converge', 'outputs/simulation')


