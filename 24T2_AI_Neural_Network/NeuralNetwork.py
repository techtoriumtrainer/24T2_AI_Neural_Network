import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump
from joblib import load
import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout


# Command to activate venv
# .venv\scripts\activate
#Import the dataset
data=pd.read_excel('Car_Purchasing_Data.xlsx')

#Create the input dataset from the original dataset by dropping the irrelevant features
# store input variables in X
X= data.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'],axis=1)

#Create the output dataset from the original dataset.
# store output variable in Y
Y= data['Car Purchase Amount']

# #Transform the input dataset into a percentage based weighted value between 0 and 1.
sc= MinMaxScaler()
X_scaled=sc.fit_transform(X)

# #Transform the output dataset into a percentage based weighted value between 0 and 1
sc1= MinMaxScaler()
y_reshape= Y.values.reshape(-1,1)
y_scaled=sc1.fit_transform(y_reshape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

#Import and Initialize the Models
lr = LinearRegression() 
svm = SVR() 
rf = RandomForestRegressor() 
gbr = GradientBoostingRegressor() 
xg =  XGBRegressor() 
dtr = DecisionTreeRegressor()
abr = AdaBoostRegressor()
etr = ExtraTreesRegressor()
ls = Lasso()
rg = Ridge()

#train the models using training sets
lr.fit(X_train,y_train)
svm.fit(X_train,y_train)
rf.fit(X_train,y_train)
gbr.fit(X_train,y_train) 
xg.fit(X_train,y_train)
dtr.fit(X_train,y_train)
abr.fit(X_train,y_train)
etr.fit(X_train,y_train)
ls.fit(X_train,y_train)
rg.fit(X_train,y_train)


# Create the neural network model with an Input layer
nn = Sequential()
nn.add(Dense(64, input_dim=5, activation='relu'))
nn.add(Dense(32, activation='relu'))
nn.add(Dense(16, activation='relu'))
nn.add(Dense(8, activation='relu'))
nn.add(Dense(1, activation='linear'))

# nn = Sequential()
# nn.add(Dense(64, input_dim=5),LeakyReLU(alpha=0.1))
# nn.add(Dense(32),ELU(alpha=1.0))
# nn.add(Dense(16, activation='relu',kernel_regularizer=l2(0.01)))
# nn.add(Dense(8, activation='relu',kernel_regularizer=l2(0.01)))
# nn.add(Dense(1, activation='linear'))


# Compile the model
nn.compile(optimizer='adam', loss='mean_squared_error')

# # Implement learning rate scheduler
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# # Early stopping to prevent overfitting
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
nn.fit(X_train, y_train, epochs=220, batch_size=25, validation_split=0.2, verbose=1)

#Prediction on the Validation/Test Data
lr_preds = lr.predict(X_test)
svm_preds = svm.predict(X_test)
rf_preds = rf.predict(X_test)
gbr_preds = gbr.predict(X_test)
xg_preds = xg.predict(X_test)
dtr_preds = dtr.predict(X_test)
abr_preds = abr.predict(X_test)
etr_preds = etr.predict(X_test)
ls_preds = ls.predict(X_test)
rg_preds = rg.predict(X_test)
nn_preds = nn.predict(X_test)

#Evaluate model performance
#RMSE is a measure of the differences between the predicted values by the model and the actual values
lr_rmse = mean_squared_error(y_test, lr_preds, squared=False)
svm_rmse = mean_squared_error(y_test, svm_preds, squared=False)
rf_rmse = mean_squared_error(y_test, rf_preds, squared=False)
gbr_rmse = mean_squared_error(y_test, gbr_preds, squared=False)
xg_rmse = mean_squared_error(y_test, xg_preds, squared=False)
dtr_rmse = mean_squared_error(y_test, dtr_preds, squared=False)
abr_rmse = mean_squared_error(y_test, abr_preds, squared=False)
etr_rmse = mean_squared_error(y_test, etr_preds, squared=False)
ls_rmse = mean_squared_error(y_test, ls_preds, squared=False)
rg_rmse = mean_squared_error(y_test, rg_preds, squared=False)
nn_rmse = mean_squared_error(y_test, nn_preds, squared=False)

#Display the evaluation results
print(f"Linear Regression RMSE: {lr_rmse}")
print(f"Support Vector Machine RMSE: {svm_rmse}")
print(f"Random Forest RMSE: {rf_rmse}")
print(f"Gradient Boosting Regressor RMSE: {gbr_rmse}")
print(f"XGBRegressor RMSE: {xg_rmse}")
print(f"Neural Network RMSE: {nn_rmse}")

#choose the best model
model_objects = [lr, svm, rf, gbr, xg, dtr, abr, etr, ls, rg, nn]
rmse_values = [lr_rmse, svm_rmse, rf_rmse, gbr_rmse, xg_rmse, dtr_rmse, abr_rmse, etr_rmse, ls_rmse, rg_rmse, nn_rmse]

best_model_index = rmse_values.index(min(rmse_values))
best_model_object = model_objects[best_model_index]

#print(f"The best model is {models[best_model_index]} with RMSE: {rmse_values[best_model_index]}")

#visualize the models results
# Create a bar chart
models = ['Linear Regression', 'Support Vector Machine', 'Random Forest', 'Gradient Boosting Regressor', 'XGBRegressor', 'DecisionTreeRegressor', 'AdaBoostRegressor', 'ExtraTreesRegressor', 'Lasso', 'Ridge', 'Neural Network']
plt.figure(figsize=(10,7))
bars = plt.bar(models, rmse_values, color=['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'grey','pink' ])


# Add RMSE values on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.00001, round(yval, 5), ha='center', va='bottom', fontsize=10)

plt.xlabel('Models')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('Model RMSE Comparison')
plt.xticks(rotation=45)  # Rotate model names for better visibility
plt.tight_layout()
# Display the chart
plt.show()

#Retrain the model on entire dataset
#nn_final = nn()
nn.fit(X_scaled, y_scaled)

# to know the internal working [optional]
#print("Coefficients:", lr_final.coef_)
#print("Intercept:", lr_final.intercept_)

#Save the Model
dump(nn, "car_model.joblib")
#Load the model
loaded_model = load("car_model.joblib")
print("Type of Model",type(loaded_model)) #Make sure the output is model type such as LinearRegression

# # Gather user inputs
gender = int(input("Enter gender (0 for female, 1 for male): "))
age = int(input("Enter age: "))
annual_salary = float(input("Enter annual salary: "))
credit_card_debt = float(input("Enter credit card debt: "))
net_worth = float(input("Enter net worth: "))

# #use the model to make predictions
X_test1= sc.transform([[gender, age, annual_salary, credit_card_debt, net_worth]])
#print(X_test1) # print just to see whether values been transformed

# #Predict on new test data
pred_value= loaded_model.predict(X_test1)
print(pred_value)
print("Predicted Car_Purchase_Amount based on input:",sc1.inverse_transform(pred_value))


