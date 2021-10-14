"""
Processing Voronoi Dataset
======================
"""

import pyspark.sql.functions as F
from pyspark.sql.functions import regexp_replace
from pyspark.sql.types import DoubleType
from cdr_trajectories.constants import spark

Voronoi_File = 'data/voronoi/*'

class Voronoi:

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
            .withColumn('avg_X', regexp_replace('avg_X', ',', '.').cast(DoubleType())) \
            .withColumn('avg_Y', regexp_replace('avg_Y', ',', '.').cast(DoubleType())) \
            .withColumn('avg_X', F.round('avg_X', 4)) \
            .withColumn('avg_Y', F.round('avg_Y', 4)) \
            .withColumnRenamed('Input_FID', 'voronoi_id')\
            .select('voronoi_id', 'avg_X', 'avg_Y')
        return self.df


if __name__ == "__main__":
    voronoi_data = Voronoi(Voronoi_File).process()
    print(voronoi_data.head())
    print(voronoi_data.count())