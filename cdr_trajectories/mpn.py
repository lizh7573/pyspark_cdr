"""
Processing MPN Dataset
======================
"""

import pyspark.sql.functions as F
from pyspark.sql.functions import regexp_replace
from pyspark.sql.types import DoubleType
from cdr_trajectories.constants import spark

MPN_File = 'data/mpn/*'

class MPN:

    def __init__(self, path):
        self.path = path
        self.df = self.read()

    def read(self):
        df = spark.read.format("csv")\
            .option("inferSchema", "true")\
            .option("header", "true")\
            .option("sep", ";")\
            .load(self.path)
        return df

    def process(self):
        self.df = self.df\
             .select('uid', 'USAGE_DTTM', 'avg_X', 'avg_Y') \
             .withColumn('avg_X', regexp_replace('avg_X', ',', '.').cast(DoubleType())) \
             .withColumn('avg_Y', regexp_replace('avg_Y', ',', '.').cast(DoubleType())) \
             .withColumn('avg_X', F.round('avg_X', 4)) \
             .withColumn('avg_Y', F.round('avg_Y', 4)) \
             .withColumnRenamed('uid', 'user_id') \
             .withColumnRenamed('USAGE_DTTM', 'timestamp') \
             .dropDuplicates(['user_id', 'timestamp']) \
             .orderBy(['user_id', 'timestamp'])
        return self.df


if __name__ == "__main__":
    mpn_data = MPN(MPN_File).process()
    print(mpn_data.head())
    print(mpn_data.count())

