# Databricks notebook source
df3=spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("dbfs:/FileStore/tables/supply_chain_data.csv")

# COMMAND ----------

df3.show(10)
