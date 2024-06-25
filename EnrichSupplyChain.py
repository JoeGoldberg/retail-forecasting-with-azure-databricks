# Databricks notebook source
# MAGIC %run ./RemoveDuplicateRowsSupplyChain

# COMMAND ----------

# MAGIC %run ./RemoveDuplicateRowsLocation

# COMMAND ----------

# MAGIC %run ./RemoveDuplicateRowsProduct

# COMMAND ----------


#Enrich SupplyChain
enrichedSupplyChain = df333.join(df666, ["LocationID"], "inner")
enrichedSupplyChain1 = enrichedSupplyChain.join(df444, ["SKU"], "inner").drop(df4["Category"])
