# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.functions import col, column
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType,StructField,StringType,IntegerType,Row

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression

# COMMAND ----------

def monToMonthText(m):
    if m == 1:
        return "Jan"
    if m == 2:
        return "Feb"
    if m == 3:
        return "Mar"
    if m == 4:
        return "Apr"
    if m == 5:
        return "May"
    if m == 6:
        return "Jun"
    if m == 7:
        return "Jul"
    if m == 8:
        return "Aug"
    if m == 9:
        return "Sept"
    if m == 10:
        return "Oct"
    if m == 11:
        return "Nov"
    if m == 12:
        return "Dec"

def monToQtr(m):
    if m >= 1 and m <= 3:
        return "Q1"
    if m >= 4 and m <= 6:
        return "Q2"
    if m >= 7 and m <= 9:
        return "Q3"
    if m >= 10 and m <= 12:
        return "Q4"

def timeOfDay(hour_int):
    if hour_int>=6 and hour_int<=10:
        tod="Morning"
    if hour_int>=11 and hour_int<=14:
        tod="Mid Day"
    if hour_int>=14 and hour_int<=18:
        tod="Afternoon"
    if hour_int>=19 and hour_int<=22:
        tod="Evening"
    return tod

getMonth = udf(lambda x: monToMonthText(x), StringType())
getQtr = udf(lambda x: monToQtr(x), StringType())
getTimeOfDay = udf(lambda x: timeOfDay(x), StringType())


# COMMAND ----------

df1=spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("dbfs:/FileStore/tables/POS.csv")
df2=spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("dbfs:/FileStore/tables/Inventory.csv")
df3=spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("dbfs:/FileStore/tables/supply_chain_data.csv")
df4=spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("dbfs:/FileStore/tables/Product.csv")
df5=spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("dbfs:/FileStore/tables/Store.csv")
df6=spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("dbfs:/FileStore/tables/Location.csv")


# COMMAND ----------

# Remove Rows with Null Values in Certain Columns
df11 = df1.na.drop(subset=["StoreID", "SKU", "Quantity", "Date", "Time"])
df22 = df2.na.drop(subset=["SKU", "StoreID", "SupplierID"])
df33 = df3.na.drop(subset=["SKU", "StoreID", "SupplierID", "LocationID"])
df44 = df4.na.drop(subset=["SKU", "ProductName", "Category"])
df55 = df5.na.drop(subset=["StoreID", "LocationID", "Address"])
df66 = df6.na.drop(subset=["Location", "Region", "State", "Zip", "Country"])

# COMMAND ----------

# Remove Duplicate Rows with Certain Columns
df111 = df11.dropDuplicates(["StoreID", "SKU", "Quantity", "Date", "Time"])
df222 = df22.dropDuplicates(["SKU", "ProductName", "StoreID", "SupplierID"])
df333 = df33.dropDuplicates(["SKU", "StoreID", "SupplierID", "LocationID"])
df444 = df44.dropDuplicates(["SKU", "ProductName", "Category"])
df555 = df55.dropDuplicates(["StoreID", "LocationID", "Address"])
df666 = df66.dropDuplicates(["Location", "Region", "State", "Zip", "Country"])


# COMMAND ----------

#Normaize Date & Time
temp11 = df111.select("*").withColumn ( 'dateInFormat', F.concat ( F.split('Date', '-')[0], F.lit('-'), F.split('Date', '-')[1], F.lit('-'),F.split('Date', '-')[2] ) ) . withColumn('Day', (F.split('Date', '-')[2]).cast('int')).withColumn('Year', F.split(col('Date'), '-')[0]). withColumn('Month', F.split('Date', '-')[1])
temp12 = temp11.select("*").withColumn('Qtr', getQtr( col('Month').cast('int'))).withColumn('Mon', getMonth( col('Month').cast('int')))
temp13 = temp12.select("*").withColumn('T', (F.split('Time', ' ')[1]))
temp14 = temp13.select("*").withColumn('H', (F.split('T', ':')[0]).cast('int'))
temp15 = temp14.select("*").withColumn('HourOfDay', getTimeOfDay(col('H').cast('int')))
temp16 = temp15.select("*").withColumn('DayOfWeek', F.date_format('dateInFormat', 'E'))
df1111 = temp16
df111.show(10)
#temp11.show(10)
#temp12.show(10)
temp16.show(10)


# COMMAND ----------

#Enrich POS Data
enrichedPOSStep1 = df1111.join(df5, ["StoreID"])
enrichedPOSStep2 = enrichedPOSStep1.join(df666, ["LocationID"])


# COMMAND ----------

df1111.show(10)

# COMMAND ----------

#Enrich SupplyChain
enrichedSupplyChain = df333.join(df6, ["LocationID"], "inner")
enrichedSupplyChain1 = enrichedSupplyChain.join(df444, ["SKU"], "inner").drop(df4["Category"])


# COMMAND ----------

#RandomForestRegressor - Inventory Prediction
cat_cols = ["Category", "StoreID", "SupplierID", "LocationID", "Region", "State", "Zip", "ProductName"]
stages = []

for c in cat_cols:
	stringIndexer = StringIndexer(inputCol=c, outputCol=c + "_index")
	encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()],outputCols=[c + "_vec"])
	stages += [stringIndexer, encoder]


# Transform all features into a vector
num_cols = ["Costs", "ProductionVolumes"]
assemblerInputs = [c + "_vec" for c in cat_cols] + num_cols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# Create pipeline and use on dataset
pipeline = Pipeline(stages=stages)
enrichedSupplyChain3 = pipeline.fit(enrichedSupplyChain1).transform(enrichedSupplyChain1)

train, test = enrichedSupplyChain3.randomSplit([0.80, 0.20], seed=12345)

rf = RandomForestRegressor(featuresCol='features', labelCol='NumberOfProductsSold')
rf_model = rf.fit(train)

train_predictions = rf_model.transform(train)
test_preds = rf_model.transform(test)
print(test_preds)

#*** NEED TO CORRECT THIS
#test_preds.select("Category", "StoreID", "SupplierID", "LocationID", "Region", "State", "Zip", "ProductName", "prediction").write.mode("overwrite").csv("///G://My Drive//Sumit_Consulting//2023//Clients//BMC//Project2//HandsOnProj1//Code//SumitData//RandomForestRegressorInventory_tests_preds.csv")
test_preds.write.mode("overwrite").parquet("dbfs:/FileStore/tables/RandomForestRegressorInventory_tests_preds.parquet")

def extract_feature_imp(feature_imp, dataset, features_col):
    list_extract = []
    for i in dataset.schema[features_col].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[features_col].metadata["ml_attr"]["attrs"][i]
    feature_list = pd.DataFrame(list_extract)
    feature_list['score'] = feature_list['idx'].apply(lambda x: feature_imp[x])
    return(feature_list.sort_values('score', ascending = False))


feature_list = extract_feature_imp(rf_model.featureImportances, train, "features")
top_20_features = feature_list.sort_values('score', ascending = False).head(20)
print(top_20_features)
#*** NEED TO CORRECT THIS
top_20_features.to_csv('/dbfs/FileStore/tables/RandomForestRegressorInventory_Top20Features.csv', index=False)

evaluator1 = RegressionEvaluator(predictionCol="prediction",  labelCol='NumberOfProductsSold', metricName="r2")
print("Train R2:", evaluator1.evaluate(train_predictions))
print("Test R2:", evaluator1.evaluate(test_preds))


evaluator2 = RegressionEvaluator(predictionCol="prediction",  labelCol='NumberOfProductsSold', metricName="mae")
print("Train mae:", evaluator2.evaluate(train_predictions))
print("Test mae:", evaluator2.evaluate(test_preds))


evaluator3 = RegressionEvaluator(predictionCol="prediction",  labelCol='NumberOfProductsSold', metricName="rmse")
print("Train RMSE:", evaluator3.evaluate(train_predictions))
print("Test RMSE:", evaluator3.evaluate(test_preds))


# COMMAND ----------

#RandomForestRegressor - POS Prediction
cat_cols = ["SKU", "Category", "StoreID", "Year", "Mon", "Qtr", "HourOfDay", "DayOfWeek", "Location", "Region", "State", "Zip"]
stages = []

for c in cat_cols:
	stringIndexer = StringIndexer(inputCol=c, outputCol=c + "_index")
	encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()],outputCols=[c + "_vec"])
	stages += [stringIndexer, encoder]


# Transform all features into a vector
num_cols = []
assemblerInputs = [c + "_vec" for c in cat_cols] + num_cols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# Create pipeline and use on dataset
pipeline = Pipeline(stages=stages)
df1_x = pipeline.fit(enrichedPOSStep2).transform(enrichedPOSStep2)

train, test = df1_x.randomSplit([0.80, 0.20], seed=12345)

rf = RandomForestRegressor(featuresCol='features', labelCol='Quantity')
rf_model = rf.fit(train)

train_predictions = rf_model.transform(train)

test_preds = rf_model.transform(test)

print(test_preds)

#*** NEED TO CORRECT THIS
#test_preds.select("Category", "StoreID", "SupplierID", "LocationID", "Region", "State", "Zip", "ProductName", "prediction").write.mode("overwrite").csv("///G://My Drive//Sumit_Consulting//2023//Clients//BMC//Project2//HandsOnProj1//Code//SumitData//RandomForestRegressorInventory_tests_preds.csv")
test_preds.write.mode("overwrite").parquet("dbfs:/FileStore/tables/RandomForestRegressorPOS_tests_preds.parquet")


feature_list = extract_feature_imp(rf_model.featureImportances, train, "features")
top_20_features = feature_list.sort_values('score', ascending = False).head(20)
#*** NEED TO CORRECT THIS
top_20_features.to_csv('/dbfs/FileStore/tables/RandomForestRegressorPOS_Top20Features.csv', index=False)

# Then make your desired plot function to visualize feature importance
#plot_feature_importance(top_20_features['score'], top_20_features['name'])


# Evaluation
evaluator1 = RegressionEvaluator(predictionCol="prediction",  labelCol='Quantity', metricName="r2")
print("Train R2:", evaluator1.evaluate(train_predictions))
print("Test R2:", evaluator1.evaluate(test_preds))


evaluator2 = RegressionEvaluator(predictionCol="prediction",  labelCol='Quantity', metricName="mae")
print("Train mae:", evaluator2.evaluate(train_predictions))
print("Test mae:", evaluator2.evaluate(test_preds))

evaluator3 = RegressionEvaluator(predictionCol="prediction",  labelCol='Quantity', metricName="rmse")
print("Train RMSE:", evaluator3.evaluate(train_predictions))
print("Test RMSE:", evaluator3.evaluate(test_preds))


# COMMAND ----------

#Linear Regression Inventory
cat_cols = ["Category", "StoreID", "SupplierID", "LocationID", "Region", "State", "Zip", "ProductName"]
stages = []

for c in cat_cols:
	stringIndexer = StringIndexer(inputCol=c, outputCol=c + "_index")
	encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()],outputCols=[c + "_vec"])
	stages += [stringIndexer, encoder]


# Transform all features into a vector
num_cols = ["Costs", "ProductionVolumes"]
assemblerInputs = [c + "_vec" for c in cat_cols] + num_cols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# Create pipeline and use on dataset
pipeline = Pipeline(stages=stages)
df3_x = pipeline.fit(enrichedSupplyChain1).transform(enrichedSupplyChain1)


train, test = df3_x.randomSplit([0.90, 0.10], seed=1234567)


# Fit scaler to train dataset
scaler = StandardScaler().setInputCol('features').setOutputCol('scaled_features')
scaler_model = scaler.fit(train)

# Scale train and test features
train = scaler_model.transform(train)
test = scaler_model.transform(test)


lr = LinearRegression(featuresCol='scaled_features', labelCol='NumberOfProductsSold')
lr_model = lr.fit(train)

train_predictions = lr_model.transform(train)
test_predictions = lr_model.transform(test)
print(test_predictions)

evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="NumberOfProductsSold", metricName="r2")
evaluator1 = RegressionEvaluator(predictionCol="prediction", labelCol="NumberOfProductsSold", metricName="mae")
evaluator2 = RegressionEvaluator(predictionCol="prediction", labelCol="NumberOfProductsSold", metricName="rmse")

print("Train R2:", evaluator.evaluate(train_predictions))
print("Test R2:", evaluator.evaluate(test_predictions))

print("Train MAE:", evaluator1.evaluate(train_predictions))
print("Test MAE:", evaluator1.evaluate(test_predictions))

print("Train RMSE:", evaluator2.evaluate(train_predictions))
print("Test RMSE:", evaluator2.evaluate(test_predictions))


print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

list_extract = []
for i in df3_x.schema['features'].metadata["ml_attr"]["attrs"]:
    list_extract = list_extract + df3_x.schema['features'].metadata["ml_attr"]["attrs"][i]

varlist = pd.DataFrame(list_extract)
varlist['weight'] = varlist['idx'].apply(lambda x: lr_model.coefficients[x])
weights = varlist.sort_values('weight', ascending = False)
#*** NEED TO CORRECT THIS
weights.to_csv('/dbfs/FileStore/tables/LinearRegressionInventory_Weights.csv', index=False)


# COMMAND ----------

#Linear Regression POS
cat_cols = ["SKU", "Category", "StoreID", "Year", "Mon", "Qtr", "HourOfDay", "DayOfWeek", "Location", "Region", "State", "Zip"]
stages = []

for c in cat_cols:
	stringIndexer = StringIndexer(inputCol=c, outputCol=c + "_index")
	encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()],outputCols=[c + "_vec"])
	stages += [stringIndexer, encoder]


# Transform all features into a vector
num_cols = ["Quantity"]
assemblerInputs = [c + "_vec" for c in cat_cols] + num_cols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# Create pipeline and use on dataset
pipeline = Pipeline(stages=stages)
df3_x = pipeline.fit(enrichedPOSStep2).transform(enrichedPOSStep2)

train, test = df3_x.randomSplit([0.90, 0.10], seed=1234567)

# Fit scaler to train dataset
scaler = StandardScaler().setInputCol('features').setOutputCol('scaled_features')
scaler_model = scaler.fit(train)

# Scale train and test features
train = scaler_model.transform(train)
test = scaler_model.transform(test)


lr = LinearRegression(featuresCol='scaled_features', labelCol='Quantity')
lr_model = lr.fit(train)

train_predictions = lr_model.transform(train)
test_predictions = lr_model.transform(test)
print(test_predictions)

evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="Quantity", metricName="r2")
print("Train R2:", evaluator.evaluate(train_predictions))
print("Test R2:", evaluator.evaluate(test_predictions))
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))


evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="Quantity", metricName="mae")
print("Train MAE:", evaluator.evaluate(train_predictions))
print("Test MAE:", evaluator.evaluate(test_predictions))
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

list_extract = []
for i in df3_x.schema['features'].metadata["ml_attr"]["attrs"]:
    list_extract = list_extract + df3_x.schema['features'].metadata["ml_attr"]["attrs"][i]

varlist = pd.DataFrame(list_extract)
varlist['weight'] = varlist['idx'].apply(lambda x: lr_model.coefficients[x])
weights = varlist.sort_values('weight', ascending = False)

#*** NEED TO CORRECT THIS
weights.to_csv('/dbfs/FileStore/tables/LinearRegressionPOS_Weights.csv', index=False)
