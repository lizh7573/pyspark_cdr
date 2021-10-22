"""
Time Inhomogeneous Transition Matrix
====================================
"""


import pyspark.sql.functions as F

class time_inhomo:

    def __init__(self, df, begin, end):
        self.df = df
        self.begin = begin
        self.end = end

    def filter(self):
        self.df = self.df.filter((F.col('hour') >= self.begin) & (F.col('hour') <= self.end))
        return self.df

    def make_tm_time(self):
        self.filter()
        return self.df