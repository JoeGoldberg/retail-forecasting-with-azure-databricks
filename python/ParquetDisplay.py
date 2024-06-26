# Databricks notebook source
#df = spark.read.format('parquet').options(header=True,inferSchema=True).load("dbfs:/FileStore/tables/RandomForestRegressorInventory_tests_preds.parquet")
#df.show(1000)

df1 = spark.read.format('parquet').options(header=True,inferSchema=True).load("dbfs:/FileStore/tables/RandomForestRegressorPOS_tests_preds.parquet")
df1.show(1000)

#df1.write.csv("dbfs:/FileStore/tables/RandomForestRegressorPOS_tests_preds.csv")
