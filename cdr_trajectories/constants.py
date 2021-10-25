"""
Constants for this package
==========================
"""

from pyspark.sql import SparkSession

spark = SparkSession.builder\
    .enableHiveSupport()\
    .appName('MobilityAnalysis_NoisyData')\
    .master("local[*]")\
    .getOrCreate()


ring_fraction = [0.37, 0.37, 0.15, 0.06, 0.05]

OD_time_frame = 2*60*60


