"""
Trajectories
============
"""

import pyspark.sql.functions as F
from cdr_trajectories.mpn import mpn_data
from cdr_trajectories.voronoi import voronoi_data
from cdr_trajectories.ring import oneRing_data, twoRing_data, threeRing_data
from cdr_trajectories.time_inhomo import time_inhomo


# Deterministic Trajectories
class DetermTraj:

    def __init__(self, mpn, voronoi):
        self.mpn = mpn
        self.voronoi = voronoi

    def join(self):
        self.df = self.mpn.join(self.voronoi, ['avg_X', 'avg_Y'], how = 'inner')\
                       .orderBy(['user_id', 'timestamp'])
        return self.df
            
    def process(self):
        self.df = self.df.withColumn('neighbors', F.array('voronoi_id'))\
                         .withColumn('props', F.array(F.lit(1.0)))\
                         .withColumn('states', F.arrays_zip('neighbors', 'props'))\
                         .orderBy(['user_id', 'timestamp'])\
                         .drop('avg_X', 'avg_Y', 'neighbors', 'props')
        return self.df

    def make_traj(self):
        self.join()
        self.process()
        return self.df


trajectories = DetermTraj(mpn_data, voronoi_data).join()
zeroRing_trajectories = DetermTraj(mpn_data, voronoi_data).make_traj()
deterministic_trajectories = zeroRing_trajectories



#Probabilistic Trajectories
class ProbTraj:

    def __init__(self, df, ring):
        self.df = df
        self.ring = ring

    def join(self):
        self.df = self.df.join(self.ring, ['voronoi_id'], how = 'inner')\
                         .orderBy(['user_id', 'timestamp'])\
                         .drop('avg_X', 'avg_Y', 'neighbors', 'props')
        return self.df


oneRing_trajectories = ProbTraj(trajectories, oneRing_data).join()
twoRing_trajectories = ProbTraj(trajectories, twoRing_data).join()
threeRing_trajectories = ProbTraj(trajectories, threeRing_data).join()

probabilistic_trajectories = threeRing_trajectories



# Time-inhomogeneous Trajectories
# Paremeters are subjected to change
time_inhomo_deterministic_trajectories = time_inhomo(deterministic_trajectories, 4, 4, 6, 8).make_tm_time()
time_inhomo_probabilistic_trajectories = time_inhomo(probabilistic_trajectories, 4, 4, 17, 19).make_tm_time()
