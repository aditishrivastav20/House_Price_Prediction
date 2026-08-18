import pandas as pd
df = pd.read_csv("data.csv")
print(df.head())
df.info()
print("\nMissing values:")
print(df.isnull().sum())
print("\nShape:")
print(df.shape) # How many Row and Columns are there in the dataset
print("\nColumns:")
print(df.columns) # Exact name of Target Column
# Features And Target
X = df.drop("price", axis=1)
y = df["price"]
print("Features:")# All information except target i.e Price
print(X.columns)
print("\nTarget:")# Price
print(y.head())
print("\nData Types:")
print(X.dtypes)
# Some datatypes are numeric, while some are objects like City, Country etc.
# Lets convert the object columns to numeric..
#By One Hot Coding 
X = pd.get_dummies(X, drop_first=True)
print("\nAfter One Hot Encoding:")
print(X.head())
print("\nNew Shapes:")
print(X.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTraining Features:")
print(X_train.shape)
print("\nTesting Features:")
print(X_test.shape)
print("\nTraining Target:")
print(y_train.shape)
print("\nTesting Target:")
print(y_test.shape)
# 80% data training 
# 20% data testing 
 # Linear Regression Model (Exact Coulumns Price Prdiction )
from sklearn.linear_model import LinearRegression
#Create model 
linear_model = LinearRegression()
#Train model
linear_model.fit(X_train, y_train)
print("Linear Regression Model Trained Successfully!")

