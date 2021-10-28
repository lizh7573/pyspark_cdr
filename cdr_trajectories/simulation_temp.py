"""
Simulation
==========
"""

"""
import ast
from typing import ValuesView
import IPython.display as display
import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql import Window
import scipy.sparse as sparse
from pyspark.sql.types import IntegerType, FloatType, ArrayType
from cdr_trajectories.main import oneRing_trajectories
from cdr_trajectories.constants import spark
# from cdr_trajectories.udfs import sparse_vector

test = oneRing_trajectories
# test.printSchema()
# window = Window.partitionBy(['user_id']).orderBy('timestamp')
y = test.withColumn('v_col', F.col('states').__getitem__('neighbors'))\
        .withColumn('v_val', F.col('states').__getitem__('props'))\
        .withColumn('array_size', F.size(F.col('v_col')))\
        .withColumn('v_row', F.expr('array_repeat(0, array_size)'))\
        .select('voronoi_id', 'user_id', 'timestamp', 'v_row', 'v_col', 'v_val')
y = y.limit(2)
# y2 = y.toPandas().to_csv('temp2.csv')

def fun1(x):

    voronoi_id = x.voronoi_id
    user_id = x.user_id
    timestamp = x.timestamp
    v_row = x.v_row
    v_col = x.v_col
    v_val = x.v_val
    vector = sparse.coo_matrix((x.v_val, (x.v_row, x.v_col)), shape=(1,114)).toarray().tolist()

    return (voronoi_id, user_id, timestamp, v_row, v_col, v_val, vector)

rdd2 = y.rdd.map(lambda x: fun1(x))\
        .toDF(['voronoi_id', 'user_id', 'timestamp', 'v_row', 'v_col', 'v_val', 'vector'])
# rdd2.show()
rdd2.toPandas().to_csv('result.csv')

"""











        # .withColumn('i', F.row_number().over(window))
# df = y.toPandas()
# w = sparse.coo_matrix((df['v_val'][0], (df['v_row'][0], df['v_col'][0])), shape = (1, 114)).toarray()
# v = np.array2string(w, precision = 4, separator = ',', suppress_small = True)

# df['vector'] = v
# df['vector'] = df['vector'].apply(lambda s: ast.literal_eval(s))

# print(df)
# df.to_csv('temp1.csv')
# spark.createDataFrame(df).printSchema()

# window = Window.partitionBy(['user_id', 'timestamp'])
# y1 = y.withColumn('zip', F.arrays_zip('v_row', 'v_col', 'v_val'))\
#        .withColumn('zip', F.explode('zip'))\
#        .select('voronoi_id', 'timestamp', 'weekday', 'hour',
#                F.col('zip.v_row').alias('v_row'),
#                F.col('zip.v_col').alias('v_col'),
#                F.col('zip.v_val').alias('v_val'))

# df = y1.toPandas()
# df.to_csv('temp1.csv')
# df['vector'] = sparse.coo_matrix((vals, (rows, cols)), shape = (1, 114)).toarray()


# def sparse_vector(data, val):

#     pd_df = data.toPandas()

#     rows = np.array( pd_df['v_row'] )
#     cols = np.array( pd_df['v_col'])
#     vals = np.array( pd_df[val] )

#     M = sparse.coo_matrix((vals, (rows, cols)), shape = (1, 114)).toarray().tolist()
    # V = spark.createDataFrame(pd.DataFrame({'vector1': M[0, :]}))

    # return M

# window = Window.partitionBy(['user_id', 'timestamp'])
# sparse_vector_udf = F.udf(sparse_vector, ArrayType(ArrayType(FloatType())))
# print(sparse_vector_udf)
# print(type(sparse_vector_udf))
# y2 = y1.withColumn('vector', sparse_vector_udf(y1, F.col('v_val')).over(window))\
#        .dropDuplicates(['user_id', 'timestamp'])



# pd_df = y1.toPandas()

# rows = np.array( pd_df['v_row'] )
# cols = np.array( pd_df['v_col'])
# vals = np.array( pd_df['v_val'] )

# M = sparse.coo_matrix((vals, (rows, cols)), shape = (1, 114)).toarray()


# print(M)
# print(V)
# print(type(V))



    






    
