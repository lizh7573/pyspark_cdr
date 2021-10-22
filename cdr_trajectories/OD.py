"""
Origin-Destination Matrix
=========================
"""

import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.types import ArrayType, FloatType, LongType
from cdr_trajectories.udfs import matrix_updates
from cdr_trajectories.constants import OD_time_frame

class OD:

    def __init__(self, df):
        self.df = df

    def od_states_vector(self):
        window = Window.partitionBy(['user_id', F.to_date('timestamp')])\
                       .orderBy('timestamp_long').rangeBetween(-OD_time_frame, OD_time_frame)
        self.df = self.df.withColumn('timestamp_long', F.col('timestamp').cast(LongType()))\
                         .withColumn('first_idx', F.first('voronoi_id').over(window))\
                         .withColumn('first_ts', F.first('timestamp').over(window))\
                         .withColumn('last_idx', F.last('voronoi_id').over(window))\
                         .withColumn('last_ts', F.last('timestamp').over(window))\
                         .withColumn('last_states', F.last('states').over(window))
        return self.df

    def od_states_update(self):
        updates_udf = F.udf(matrix_updates, ArrayType(ArrayType(FloatType())))
        self.df = self.df.withColumn('updates', updates_udf('states', 'last_states'))
        return self.df

    def states_collect(self):
        self.df = self.df.select(['updates'])\
                    .withColumn('updates', F.explode('updates'))\
                    .withColumn('y', F.col('updates').getItem(0))\
                    .withColumn('x', F.col('updates').getItem(1))\
                    .withColumn('val', F.col('updates').getItem(2))\
                    .drop('updates')\
                    .groupBy(['y', 'x']).agg(F.sum('val').alias('updates'))
        return self.df

    def states_normalize(self):
        window = Window.partitionBy(F.col('y'))
        self.df = self.df.withColumn('updates', F.col('updates')/F.sum(F.col('updates')).over(window))
        return self.df

    def make_od(self):
        self.od_states_vector()
        self.od_states_update()
        self.states_collect()
        self.states_normalize()
        return self.df

    def make_od_dataset(self):
        self.od_states_vector()
        self.od_states_update()
        return self.df