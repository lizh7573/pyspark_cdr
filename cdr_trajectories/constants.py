"""
Constants for This Package
==========================
"""

from pyspark.sql import SparkSession

spark = SparkSession.builder\
    .enableHiveSupport()\
    .appName('cdr_trajectories')\
    .master("local[*]")\
    .getOrCreate()


ring_fraction = [0.37, 0.37, 0.15, 0.06, 0.05]

OD_time_frame = 2*60*60


