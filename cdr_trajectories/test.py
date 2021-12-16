
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import Window
# from cdr_trajectories.constants import Spark
from pyspark.sql import Row
from pyspark.sql.types import ArrayType, IntegerType, StringType, FloatType

spark = SparkSession.builder\
    .enableHiveSupport()\
    .appName('cdr_trajectories')\
    .getOrCreate()



test = spark.range(1, 11)\
            .withColumn('rand', F.round(F.rand(seed=0)*114, 0).cast(IntegerType()))

test.show()