# Databricks notebook source
df5=spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("dbfs:/FileStore/tables/Store.csv")
