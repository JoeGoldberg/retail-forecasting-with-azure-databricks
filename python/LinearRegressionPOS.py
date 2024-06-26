# Databricks notebook source
# MAGIC %run ./AllImports

# COMMAND ----------

# MAGIC %run ./EnrichPOSData

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

weights.to_csv('/dbfs/FileStore/tables/LinearRegressionPOS_Weights.csv', index=False)
