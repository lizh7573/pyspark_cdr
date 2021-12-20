
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



test = spark.range(1, 4)
            # .withColumn('rand', F.round(F.rand(seed=0)*114, 0).cast(IntegerType()))

def rand(data, n):

    df = data

    for i in range(n-1):
        df = df.union(data)

    window = Window.partitionBy(['id']).orderBy(F.lit('A'))

    df = df.withColumn('i', F.row_number().over(window))\
           .withColumn('rand', F.round(F.rand(seed=0)*114, 0).cast(IntegerType()))\
           .withColumnRenamed('id', 'user_id')

    return df

test2 = rand(test, 5)

test2.show()

