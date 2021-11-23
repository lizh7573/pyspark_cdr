

import pyspark.sql.functions as F
from cdr_trajectories.constants import Spark

mpn_file = 'data/mpn/*'

Spark.conf.set('spark.sql.shuffle.partitions')

df_mpn = Spark.read.format("csv")\
              .option("inferSchema", "true")\
              .option("header", "true")\
              .option("sep", ";")\
              .load(mpn_file)

df_mpn.write.mode('overwrite').parquet('df_mpn.parquet')

parquet_mpn = Spark.read.parquet('df_mpn.parquet')

print()