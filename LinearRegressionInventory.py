# Databricks notebook source
# MAGIC %run ./AllImports

# COMMAND ----------

# MAGIC %run ./EnrichSupplyChain

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
weights.to_csv('/dbfs/FileStore/tables/LinearRegressionInventory_Weights.csv', index=False)
