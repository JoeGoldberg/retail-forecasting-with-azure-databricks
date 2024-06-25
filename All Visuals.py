# Databricks notebook source
# Import the necessary modules 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns

plt.figure(figsize=(10,8))  
# Initialize the lists for X and Y 
data = pd.read_csv('/dbfs/FileStore/tables/LinearRegressionInventory_Weights.csv') 
  
df = pd.DataFrame(data) 
 
Y = list(df.iloc[:, 1]) 
X = list(df.iloc[:, 2]) 
  
# Plot the data using bar() method 
#plt.bar(X, Y, color='g') 
plt.barh(Y, X, color='g') 
#plt.title("Linerar Regression Inventory Feature Weight") 
plt.xlabel("Weight") 
plt.ylabel("Features") 

sns.barplot(y=Y,x=X).set(title='Linear Regression Inventory Feature Weight')
# Show the plot 
plt.show() 


# COMMAND ----------

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(10,8))
data = pd.read_csv('/dbfs/FileStore/tables/LinearRegressionPOS_Weights.csv')
data1 = data.loc[data["name"] != "Quantity"]
#display(data)
  
df = pd.DataFrame(data1) 
  
Y = list(df.iloc[:, 1]) 
X = list(df.iloc[:, 2]) 
plt.xlabel("Weight") 
plt.ylabel("Features")
sns.barplot(y=Y,x=X).set(title='Linear Regression POS Feature Weight')
plt.show()

# COMMAND ----------

# Import the necessary modules 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 

plt.figure(figsize=(10,8))
# Initialize the lists for X and Y 
data = pd.read_csv('/dbfs/FileStore/tables/RandomForestRegressorInventory_Top20Features.csv') 
  
df = pd.DataFrame(data) 
  
Y = list(df.iloc[:, 1]) 
X = list(df.iloc[:, 2]) 
  
# Plot the data using bar() method 
#plt.bar(X, Y, color='g') 
#plt.barh(Y, X, color='g') 
#plt.title("Random Forest Inventory Top 20 by Importance") 
plt.xlabel("Weight") 
plt.ylabel("Features") 

sns.barplot(y=Y,x=X).set(title='Random Forest Inventory Top 20 Features by Importance')  
# Show the plot 
plt.show() 


# COMMAND ----------

# Import the necessary modules 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 

plt.figure(figsize=(10,8))

# Initialize the lists for X and Y 
data = pd.read_csv('/dbfs/FileStore/tables/RandomForestRegressorPOS_Top20Features.csv') 
  
df = pd.DataFrame(data) 
  
Y = list(df.iloc[:, 1]) 
X = list(df.iloc[:, 2]) 
  
# Plot the data using bar() method 
#plt.bar(X, Y, color='g') 
#plt.barh(Y, X, color='g') 
#plt.title("Random Forest POS Feature Top 20 by Importance") 
plt.xlabel("Weight") 
plt.ylabel("Features") 

sns.barplot(y=Y,x=X).set(title='Random Forest POS Top 20 Features by Importance')  

# Show the plot 
plt.show() 
