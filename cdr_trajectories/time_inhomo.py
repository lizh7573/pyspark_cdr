"""
Time Inhomogeneous Transition Matrix
====================================
"""


import pyspark.sql.functions as F

class time_inhomo:

    def __init__(self, df, day_begin, day_end, hour_begin, hour_end):
        self.df = df
        self.day_begin = day_begin
        self.day_end = day_end
        self.hour_begin = hour_begin
        self.hour_end = hour_end

    def filter(self):
        self.df = self.df\
            .filter((F.col('weekday') >= self.day_begin) & (F.col('weekday') <= self.day_end))\
            .filter((F.col('hour') >= self.hour_begin) & (F.col('hour') <= self.hour_end))
        return self.df

    def make_tm_time(self):
        self.filter()
        return self.df