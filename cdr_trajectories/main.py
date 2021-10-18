"""
Main module
===========
"""

from cdr_trajectories.mpn import mpn_data
from cdr_trajectories.voronoi import voronoi_data
from cdr_trajectories.ring import ring_data

trajectories = mpn_data.join(voronoi_data, ['avg_X', 'avg_Y'], how = 'inner')\
                       .orderBy(['user_id', 'timestamp'])

cdr_trajectories = trajectories.join(ring_data, ['voronoi_id'], how = 'inner')\
                               .orderBy(['user_id', 'timestamp'])\
                               .drop('avg_X', 'avg_Y', 'neighbors', 'props')

print(cdr_trajectories.show(10))


