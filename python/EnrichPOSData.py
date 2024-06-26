# Databricks notebook source
# MAGIC %run ./NormalizeDateTimePOS

# COMMAND ----------

# MAGIC %run ./RemoveDuplicateRowsStores

# COMMAND ----------

# MAGIC %run ./RemoveDuplicateRowsLocation

# COMMAND ----------

#Enrich POS Data
enrichedPOSStep1 = df1111.join(df555, ["StoreID"])
enrichedPOSStep2 = enrichedPOSStep1.join(df666, ["LocationID"])
