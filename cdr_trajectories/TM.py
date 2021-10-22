"""
Transition Matrix
=================
"""

import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.types import ArrayType, FloatType
from cdr_trajectories.udfs import matrix_updates

class TM:

    def __init__(self, df):
        self.df = df

    def tm_states_vector(self):
        window = Window.partitionBy(['user_id', F.to_date('timestamp')]).orderBy('timestamp')
        self.df = self.df.withColumn('states_lag', F.lag('states').over(window)).dropna()
        return self.df

    def tm_states_update(self):
        updates_udf = F.udf(matrix_updates, ArrayType(ArrayType(FloatType())))
        self.df = self.df.withColumn('updates', updates_udf('states_lag', 'states'))
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

    def make_tm(self):
        self.tm_states_vector()
        self.tm_states_update()
        self.states_collect()
        self.states_normalize()
        return self.df

    

    

    