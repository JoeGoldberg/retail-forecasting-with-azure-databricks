# Databricks notebook source
df4=spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("dbfs:/FileStore/tables/Product.csv")
