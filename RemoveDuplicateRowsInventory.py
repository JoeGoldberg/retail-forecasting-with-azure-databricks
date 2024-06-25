# Databricks notebook source
# MAGIC %run ./RemoveNullColumnsInventory

# COMMAND ----------

# Remove Rows with Null Values in Certain Columns
df222 = df22.dropDuplicates(["SKU", "ProductName", "StoreID", "SupplierID"])
