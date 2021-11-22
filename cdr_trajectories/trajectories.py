"""
Trajectories
============
"""

import pyspark.sql.functions as F
from cdr_trajectories.mpn import MPN
from cdr_trajectories.voronoi import Voronoi
from cdr_trajectories.ring import get_RingData


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





# Probabilistic Trajectories
class ProbTraj:

    def __init__(self, df, ring):
        self.df = df
        self.ring = ring

    def make_traj(self):
        self.df = self.df.join(self.ring, ['voronoi_id'], how = 'inner')\
                         .orderBy(['user_id', 'timestamp'])\
                         .drop('avg_X', 'avg_Y', 'neighbors', 'props')
        return self.df





def get_DetermTrajData(mpn_file, voronoi_file):

    mpn_data = MPN(mpn_file).process()
    voronoi_data = Voronoi(voronoi_file).process()
    
    deterministic_trajectories = DetermTraj(mpn_data, voronoi_data).make_traj()

    return deterministic_trajectories


def get_ProbTrajData(mpn_file, voronoi_file, firstRing_file, secondRing_file, thirdRing_file):

    mpn_data = MPN(mpn_file).process()
    voronoi_data = Voronoi(voronoi_file).process()

    trajectories = DetermTraj(mpn_data, voronoi_data).join()
    threeRing_data = get_RingData(firstRing_file, secondRing_file, thirdRing_file)

    probabilistic_trajectories = ProbTraj(trajectories, threeRing_data).make_traj()

    return probabilistic_trajectories