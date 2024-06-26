# Databricks notebook source
# MAGIC %run ./LoadSupplyChain

# COMMAND ----------

# Remove Rows with Null Values in Certain Columns
df33 = df3.na.drop(subset=["SKU", "StoreID", "SupplierID", "LocationID"])
