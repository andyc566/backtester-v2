import os
from preprocessing import load_data, reduce_mem_usage
from model_training import train_lgb, train_xgb, train_cbt, save_model
from inference import predict
import pandas as pd
from sklearn.model_selection import train_test_split
    
# Define paths
data_path = "./data/example_data.xlsx"
model_dir = "./models/"

# Load and preprocess data
df = load_data(data_path)
df = reduce_mem_usage(df)

# Split data
X = df.drop(columns=["target", "weight"])
y = df["target"]
weights = df["weight"]

X_train, X_valid, y_train, y_valid, w_train, w_valid = train_test_split(
    X, y, weights, test_size=0.2, random_state=42
)

# Train models
lgb_model = train_lgb(X_train, y_train, X_valid, y_valid, w_train, w_valid)
save_model(lgb_model, f"{model_dir}/lgb_model.joblib")

xgb_model = train_xgb(X_train, y_train, X_valid, y_valid, w_train, w_valid)
save_model(xgb_model, f"{model_dir}/xgb_model.joblib")

cbt_model = train_cbt(X_train, y_train, X_valid, y_valid, w_train, w_valid)
save_model(cbt_model, f"{model_dir}/cbt_model.joblib")

# Make predictions (example)
X_test = X_valid  # Replace with actual test data
models = [lgb_model, xgb_model, cbt_model]
predictions = predict(models, X_test)

print(predictions)
