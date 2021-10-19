"""
Transition Matrix
=================
"""

import pyspark.sql.functions as F
from pyspark.sql import Window
from pyspark.sql.types import ArrayType, FloatType
from cdr_trajectories.udfs import transition_matrix_updates

class TM:

    def __init__(self, df):
        self.df = df

    def states_vector(self):
        window = Window.partitionBy(['user_id', F.to_date('timestamp')]).orderBy('timestamp')
        self.df = self.df.withColumn('states_lag', F.lag('states').over(window)).dropna()
        return self.df

    def states_update(self):
        tm_updates_udf = F.udf(transition_matrix_updates, ArrayType(ArrayType(FloatType())))
        self.df = self.df.withColumn('TM_updates', tm_updates_udf('states_lag', 'states'))
        return self.df

    def states_collect(self):
        self.df = self.df.select(['TM_updates'])\
                    .withColumn('TM_updates', F.explode('TM_updates'))\
                    .withColumn('y', F.col('TM_updates').getItem(0))\
                    .withColumn('x', F.col('TM_updates').getItem(1))\
                    .withColumn('val', F.col('TM_updates').getItem(2))\
                    .drop('TM_updates')\
                    .groupBy(['y', 'x']).agg(F.sum('val').alias('TM_updates'))
        return self.df

    def states_normalize(self):
        window = Window.partitionBy(F.col('x'))
        self.df = self.df.withColumn('TM_updates', F.col('TM_updates')/F.sum(F.col('TM_updates')).over(window))
        return self.df

    def make_tm(self):
        self.states_vector()
        self.states_update()
        self.states_collect()
        self.states_normalize()
        return self.df

    

    