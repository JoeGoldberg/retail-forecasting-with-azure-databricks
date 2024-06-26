# Databricks notebook source
df1=spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("dbfs:/FileStore/tables/POS.csv")
