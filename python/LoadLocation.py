# Databricks notebook source
df6=spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("dbfs:/FileStore/tables/Location.csv")
