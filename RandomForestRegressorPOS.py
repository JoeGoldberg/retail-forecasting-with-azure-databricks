# Databricks notebook source
# MAGIC %run ./AllImports

# COMMAND ----------

# MAGIC %run ./SetupUtilFunctions

# COMMAND ----------

# MAGIC %run ./EnrichPOSData

# COMMAND ----------

enrichedPOSStep2.show(10)

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
