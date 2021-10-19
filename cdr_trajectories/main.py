"""
Main module
===========
"""

from IPython.display import display
from cdr_trajectories.mpn import mpn_data
from cdr_trajectories.voronoi import voronoi_data
from cdr_trajectories.ring import ring_data
from cdr_trajectories.TM import TM
from cdr_trajectories.udfs import prepare_for_plot, plot

trajectories = mpn_data.join(voronoi_data, ['avg_X', 'avg_Y'], how = 'inner')\
                       .orderBy(['user_id', 'timestamp'])

cdr_trajectories = trajectories.join(ring_data, ['voronoi_id'], how = 'inner')\
                               .orderBy(['user_id', 'timestamp'])\
                               .drop('avg_X', 'avg_Y', 'neighbors', 'props')

tm = TM(cdr_trajectories).make_tm()

plot(prepare_for_plot(tm, 'TM_updates'), 'TM.png', 'Transition Matrix')



