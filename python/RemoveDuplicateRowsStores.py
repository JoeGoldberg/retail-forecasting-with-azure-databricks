# Databricks notebook source
# MAGIC %run ./RemoveNullColumnsStore

# COMMAND ----------

# Remove Rows with Null Values in Certain Columns
df555 = df55.dropDuplicates(["StoreID", "LocationID", "Address"])
