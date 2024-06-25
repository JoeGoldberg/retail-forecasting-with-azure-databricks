# Databricks notebook source
# MAGIC %run ./AllImports

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


def extract_feature_imp(feature_imp, dataset, features_col):
    list_extract = []
    for i in dataset.schema[features_col].metadata["ml_attr"]["attrs"]:
        list_extract = list_extract + dataset.schema[features_col].metadata["ml_attr"]["attrs"][i]
    feature_list = pd.DataFrame(list_extract)
    feature_list['score'] = feature_list['idx'].apply(lambda x: feature_imp[x])
    return(feature_list.sort_values('score', ascending = False))


