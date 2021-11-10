"""
Simulation
==========
"""


import pyspark.sql.functions as F
import scipy.sparse as sparse
from pyspark.sql import Window
from cdr_trajectories.constants import spark
from pyspark.sql.types import IntegerType
from cdr_trajectories.ring import threeRing_data
from cdr_trajectories.udfs import sim_vectorize



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




class Simulation:

    def __init__(self, df):
        self.df = spark.createDataFrame(df)

    def make_traj(self):
        w = Window.partitionBy(['user_id']).orderBy(F.lit('A'))
        self.df = self.df\
             .withColumn('simulated_traj', F.explode(F.split(F.col('simulated_traj'), ',')))\
             .withColumn('simulated_traj', F.regexp_replace('simulated_traj', '\\[', ''))\
             .withColumn('simulated_traj', F.regexp_replace('simulated_traj', '\\]', ''))\
             .withColumn('simulated_traj', F.col('simulated_traj').cast(IntegerType()))\
             .withColumn('i', F.row_number().over(w))\
             .select('user_id', 'simulated_traj', 'i')
        return self.df

    def join_ring(self):
        self.df = self.df\
             .join(threeRing_data, self.df.simulated_traj == threeRing_data.voronoi_id, how = 'inner')\
             .orderBy(['user_id', 'i']).select('user_id', 'simulated_traj', 'states', 'i')
        return self.df

    def set_helpCols(self):
        self.df = self.df\
             .withColumn('v_col', F.col('states').__getitem__('neighbors'))\
             .withColumn('v_val', F.col('states').__getitem__('props'))\
             .withColumn('array_size', F.size(F.col('v_col')))\
             .withColumn('v_row', F.expr('array_repeat(0, array_size)'))\
             .select('user_id', 'simulated_traj', 'v_row', 'v_col', 'v_val', 'i')
        return self.df

    def vectorization(self):
        self.df = self.df\
            .rdd.map(lambda x: sim_vectorize(x))\
            .toDF(['user_id', 'simulated_traj', 'sim_vector', 'i'])
        return self.df

    def simulate_vector(self):
        w = Window().orderBy('i') 
        self.df = self.df\
             .select(['i', 'sim_vector']).groupBy('i')\
             .agg(F.array(*[F.avg(F.col('sim_vector')[m]) for m in range(114+1)]).alias('sim_vector')).orderBy('i')\
             .withColumn('avg_sim_vector', F.array(*[F.avg(F.col('sim_vector')[n]).over(w) for n in range(114+1)]))\
             .drop('sim_vector').withColumnRenamed('avg_sim_vector', 'vector')

    def process(self):
        self.make_traj()
        self.join_ring()
        self.set_helpCols()
        self.vectorization()
        self.simulate_vector()
        return self.df

    

    



