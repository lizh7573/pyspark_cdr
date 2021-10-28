"""
Main module
===========
"""

import os
from cdr_trajectories.TM import TM
from cdr_trajectories.OD import OD
from cdr_trajectories.udfs import prepare_for_sparse_plot, plot_sparse,\
prepare_for_dense_plot, plot_dense
from cdr_trajectories.trajectories import deterministic_trajectories, oneRing_trajectories,\
twoRing_trajectories, probabilistic_trajectories, time_inhomo_deterministic_trajectories,\
time_inhomo_probabilistic_trajectories



# Deterministic Trajectories

tm_0 = TM(deterministic_trajectories).make_tm()
plot_sparse(prepare_for_sparse_plot(tm_0, 'updates'), 'TM_0.png', 
           'Transition Matrix (Deterministic)', 'outputs/determTraj')
tm_0.toPandas().to_csv(os.path.join('outputs/determTraj', 'tm_0.csv'))
deterministic_trajectories.toPandas().to_csv(os.path.join('outputs/determTraj', 'determTraj.csv'))



#Probabilistic Trajectories

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

# Deterministic
time_tm_0 = TM(time_inhomo_deterministic_trajectories).make_tm()
plot_sparse(prepare_for_sparse_plot(time_tm_0, 'updates'), 'specific_TM_0.png', 
            'Transition Matrix (Deterministic) (Thursday: 6am to 8am)', 'outputs/time_inhomo')
time_inhomo_deterministic_trajectories.toPandas().to_csv(os.path.join('outputs/time_inhomo', 'time_DetermTraj.csv'))
time_tm_0.toPandas().to_csv(os.path.join('outputs/time_inhomo', 'time_tm_0.csv'))

# Probabilistic
time_tm_3 = TM(time_inhomo_probabilistic_trajectories).make_tm()
plot_dense(prepare_for_dense_plot(time_tm_3, 'updates'), 'specific_TM_3.png', 
            'Transition Matrix (Probabilistic) (Thursday: 17pm to 19pm)', 'outputs/time_inhomo')
time_inhomo_deterministic_trajectories.toPandas().to_csv(os.path.join('outputs/time_inhomo', 'time_ProbTraj.csv'))
time_tm_3.toPandas().to_csv(os.path.join('outputs/time_inhomo', 'time_tm_3.csv'))




#Origin-Destination Matrices
od = OD(probabilistic_trajectories).make_od()
plot_dense(prepare_for_dense_plot(od, 'updates'), 'OD.png',
          'Origin-Destination Matrix', 'outputs/od')





