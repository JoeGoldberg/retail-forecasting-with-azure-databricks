# Databricks notebook source
# MAGIC %run ./AllImports
# MAGIC

# COMMAND ----------

# MAGIC %run ./SetupUtilFunctions

# COMMAND ----------

# MAGIC %run ./RemoveDuplicateRowsPOS

# COMMAND ----------

#Normalize Date & Time
temp11 = df111.select("*").withColumn ( 'dateInFormat', F.concat ( F.split('Date', '-')[0], F.lit('-'), F.split('Date', '-')[1], F.lit('-'),F.split('Date', '-')[2] ) ) . withColumn('Day', (F.split('Date', '-')[2]).cast('int')).withColumn('Year', F.split(col('Date'), '-')[0]). withColumn('Month', F.split('Date', '-')[1])
temp12 = temp11.select("*").withColumn('Qtr', getQtr( col('Month').cast('int'))).withColumn('Mon', getMonth( col('Month').cast('int')))
temp13 = temp12.select("*").withColumn('T', (F.split('Time', ' ')[1]))
temp14 = temp13.select("*").withColumn('H', (F.split('T', ':')[0]).cast('int'))
temp15 = temp14.select("*").withColumn('HourOfDay', getTimeOfDay(col('H').cast('int')))
temp16 = temp15.select("*").withColumn('DayOfWeek', F.date_format('dateInFormat', 'E'))
df1111 = temp16

# COMMAND ----------

temp16.show(100)
