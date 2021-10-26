"""
Processing MPN Dataset
======================
"""

import pyspark.sql.functions as F
from pyspark.sql.functions import regexp_replace
from pyspark.sql.types import DoubleType
from cdr_trajectories.constants import spark

mpn_file = 'data/mpn/*'

class mpn:

    def __init__(self, path):
        self.path = path
        self.df = spark.read.format("csv")\
            .option("inferSchema", "true")\
            .option("header", "true")\
            .option("sep", ";")\
            .load(self.path)

    def process(self):
        self.df = self.df\
             .withColumn('weekday', ((F.dayofweek('USAGE_DTTM')+5)%7)+1)\
             .withColumn('hour', F.hour('USAGE_DTTM'))\
             .withColumn('avg_X', regexp_replace('avg_X', ',', '.').cast(DoubleType())) \
             .withColumn('avg_Y', regexp_replace('avg_Y', ',', '.').cast(DoubleType())) \
             .withColumn('avg_X', F.round('avg_X', 4)) \
             .withColumn('avg_Y', F.round('avg_Y', 4)) \
             .withColumnRenamed('uid', 'user_id') \
             .withColumnRenamed('USAGE_DTTM', 'timestamp') \
             .select('user_id', 'timestamp', 'weekday', 'hour', 'avg_X', 'avg_Y') \
             .dropDuplicates(['user_id', 'timestamp']) \
             .orderBy(['user_id', 'timestamp'])
        return self.df

    def get_data(self):
        self.process()
        return self.df

mpn_data = mpn(mpn_file).get_data()







