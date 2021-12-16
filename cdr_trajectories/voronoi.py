"""
Processing Voronoi Dataset
==========================
"""

import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
# from cdr_trajectories.constants import Spark



class Voronoi:

    def __init__(self, spark, path):
        self.spark = spark
        self.path = path
        self.df = self.spark.read.format("csv")\
            .option("inferSchema", "true")\
            .option("header", "true")\
            .option("sep", ";")\
            .load(self.path)

    def process(self):
        self.df = self.df\
            .withColumn('avg_X', F.regexp_replace('avg_X', ',', '.').cast(DoubleType())) \
            .withColumn('avg_Y', F.regexp_replace('avg_Y', ',', '.').cast(DoubleType())) \
            .withColumn('avg_X', F.round('avg_X', 4)) \
            .withColumn('avg_Y', F.round('avg_Y', 4)) \
            .withColumnRenamed('Input_FID', 'voronoi_id')\
            .select(['voronoi_id', 'avg_X', 'avg_Y'])
        return self.df







