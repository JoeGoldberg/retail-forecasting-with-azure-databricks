# Databricks notebook source
# MAGIC %run ./AllImports

# COMMAND ----------

# MAGIC %run ./EnrichSupplyChain

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

