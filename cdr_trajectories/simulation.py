"""
Simulation
==========
"""


import pyspark.sql.functions as F
from pyspark.sql import Window
from cdr_trajectories.udfs import sim_vectorize



class Vectorization:

    def __init__(self, df):
        self.df = df


    def set_help_cols(self):
        window = Window.partitionBy(['user_id']).orderBy('timestamp')
        last_hour = self.df.agg(F.max('hour')).collect()[0][0]
        first_hour = self.df.agg(F.min('hour')).collect()[0][0]
        self.df = self.df.withColumn('n', (F.lit(last_hour)-F.lit(first_hour)+1)*12)\
                         .withColumn('i', F.row_number().over(window)).filter(F.col('i') == 1)\
                         .withColumn('states', F.explode('states'))\
                         .select('voronoi_id', 'user_id', 'n',
                                 F.col('states.neighbors').alias('v_col'), F.col('states.props').alias('v_val'))
        return self.df


    def weighted_average(self):
        count = self.df.agg(F.countDistinct('user_id')).collect()[0][0]
        self.df = self.df\
            .select('n', 'v_col', 'v_val')\
            .groupBy('n', 'v_col')\
            .agg(F.sum('v_val').alias('v_val'))\
            .withColumn('v_val', F.col('v_val')/F.lit(count))\
            .orderBy('v_col')
        return self.df

    def collect(self):
        self.df = self.df.groupBy('n')\
                      .agg(F.collect_list('v_col').alias('col'),
                           F.collect_list('v_val').alias('val'))
        return self.df

    
    def process(self):
        self.set_help_cols()
        self.weighted_average()
        self.collect()
        return self.df




class Simulation:

    def __init__(self, df):
        self.df = df

    def set_help_cols(self):
        self.df = self.df\
             .withColumn('states', F.array_sort(F.col('states')))\
             .withColumn('col', F.col('states').__getitem__('neighbors'))\
             .withColumn('val', F.col('states').__getitem__('props'))\
             .select('user_id', 'simulated_traj', 'col', 'val', 'i')
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
             .drop('i', 'sim_vector').withColumnRenamed('avg_sim_vector', 'vector')
        return self.df

    def process(self):
        self.set_help_cols()
        self.vectorization()
        self.simulate_vector()
        return self.df



    



