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
from sklearn import tree
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
#Make Predictions
y_pred = linear_model.predict(X_test)
print("\nPredicted Prices:")
print(y_pred[:10])
# Model Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print("MAE:", mae)# Prediction actual me average price se kitna durr hai 
print("Mean Squared Error:", mse)# Errors ko Squre kr ke measure krta hai Give much importance to big errors
print("R2-Score:", r2) 
# Actual / Preduction Graphical Representation
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred, alpha=0.5) 
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()
#2nd Model
from sklearn.tree import DecisionTreeRegressor
#Create Decision Tree regressor Model 
tree_model = DecisionTreeRegressor(random_state=42)
# Train Model 
tree_model.fit(X_train, y_train)
#Make Prediction 
tree_pred = tree_model.predict (X_test)
# Evaluate Model
tree_mae = mean_absolute_error(y_test ,tree_pred)
tree_r2 = r2_score(y_test, tree_pred)
print("\nDecision Tree Regression:")
print("MAE:", tree_mae)
print("R2 Score:", tree_r2)

# 3rd Model 
from sklearn.ensemble import RandomForestRegressor
#Create Model 
forest_model = RandomForestRegressor (
n_estimators=100, random_state=42)
#Train_Model 
forest_model.fit (X_train,y_train)
# Make Prediction
forest_pred = forest_model.predict(X_test)

#Evaluation 
forest_mae = mean_absolute_error(y_test, forest_pred)
forest_r2 = r2_score(y_test, forest_pred)
print("\n Random forest Regression:")
print("MAE:", forest_mae)
print("R2 Score:",forest_r2)

# Model Comparison 
print("\n Model Comparison ")
print(" Linear Regression - MAE:", mae,
      "R2:",r2)
print("Decision Tree - MAE:", tree_mae, "R2:", tree_r2)
print("Random Forest -MAE:" , forest_mae, "R2:", forest_r2)

# Feature Importance 
importances = forest_model.feature_importances_ 
for features, value in zip (X.columns,importances):
    print(features, ":", value)

# Graph 
plt.bar (X.columns, forest_model.feature_importances_)  
plt.xlabel ("Features" ) 
plt.ylabel(" Importances ")
plt.title("Features Importances - Random Forest ")
plt.xticks(rotation = 90)
plt.show()

#new_house prediction 
new_house = X.iloc[[0]]
predicted_price = forest_model.predict(new_house)
print("\n Predicted House Price:", predicted_price [0])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled[:5])
