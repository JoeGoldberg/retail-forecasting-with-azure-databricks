# Databricks notebook source
df2=spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("dbfs:/FileStore/tables/Inventory.csv")

# COMMAND ----------

df2.show(10)
