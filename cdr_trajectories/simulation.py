"""
Simulation
==========
"""


import pyspark.sql.functions as F
from pyspark.sql import Window



class Vectorization:

    def __init__(self, df):
        self.df = df

    def set_helpCols(self):
        window = Window.partitionBy(['user_id']).orderBy('timestamp')
        self.df = self.df.withColumn('v_col', F.first(F.col('states').__getitem__('neighbors')).over(window))\
                         .withColumn('v_val', F.first(F.col('states').__getitem__('props')).over(window))\
                         .withColumn('array_size', F.size(F.col('v_col')))\
                         .withColumn('v_row', F.expr('array_repeat(0, array_size)'))\
                         .withColumn('i', F.row_number().over(window))\
                         .select('voronoi_id', 'user_id', 'timestamp', 'v_row', 'v_col', 'v_val', 'i')
        return self.df
        

class Stationary:

    def __init__(self, df):
        self.df = df

    def process(self):
        self.df = self.df\
        .withColumn('time', F.date_format('timestamp', 'HH:mm:ss'))\
        .select(['time', 'vector']).groupBy('time')\
        .agg(F.array(*[F.avg(F.col('vector')[i]) for i in range(114+1)]).alias('vector'))\
        .orderBy('time')
        return self.df


