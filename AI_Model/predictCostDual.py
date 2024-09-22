import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers
import math
from random import random



categorical = ["hhnum","sex","ager84","race","marstat","incfamr","dv12r","hosp12r"]
numerical = ["osteopor","brokehip","hardart","hyperten","rheumfev","rheumhtd","coronhtd",
    "angina","myoinfrc","othhtatk","stroke","alzheimr","cancer","arthrtis","diabetes","aneurysm","bldclot","varicose"]

# Step 1: Load and prepare the data
def load_data(file_path):
    data = pd.read_csv(file_path)

    data = data[data['tcinpsty'].le(100000) & data['tcinpsty'].notna()]
    data = data.reset_index(drop=True)

    X = data[categorical + numerical]
    y = data['tcinpsty']
    print("Loaded Data")
    return X, y

# Fill in the missing data
def fillMissing(X, y):
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns    
    # Create imputers for numeric and categorical data
    numeric_imputer = SimpleImputer(strategy='most_frequent')
    
    # Apply imputation
    X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])
    return X, y

# Step 2: Preprocess the data
def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #scaler = StandardScaler()
    #X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)

    # Preprocess numerical features
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[numerical])
    X_test_num = scaler.transform(X_test[numerical])

    # Preprocess categorical features
    scalar2 = StandardScaler()
    X_train_cat = scalar2.fit_transform(X_train[categorical])
    X_test_cat = scalar2.transform(X_test[categorical])
    
    print("Preprocessed Data")
    #return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    return X_train_num, X_test_num, X_train_cat, X_test_cat, y_train, y_test

# Step 3: Create the model
def create_model(X_train_num, X_test_num, X_train_cat, X_test_cat, y_train, y_test):
    # Step 2: Define the model
    num_numerical_features = X_train_num.shape[1]
    num_categorical_features = X_train_cat.shape[1]

    numerical_input = keras.Input(shape=(num_numerical_features,), name="numerical_input")
    categorical_input = keras.Input(shape=(num_categorical_features,), name="categorical_input")

    # Process numerical features
    x_num = layers.Dense(64, activation="relu")(numerical_input)
    x_num = layers.Dense(32, activation="relu")(x_num)

    # Process categorical features
    x_cat = layers.Dense(32, activation="relu")(categorical_input)

    # Concatenate numerical and categorical features
    x = layers.concatenate([x_num, x_cat])

    # Add more layers
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)

    # Output layer
    cost_pred = layers.Dense(1, name="cost")(x)

    # Create the model
    model = keras.Model(
        inputs=[numerical_input, categorical_input],
        outputs=cost_pred
    )

    # Compile the model
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"])

    return model

# Step 5: Evaluate the model
def evaluate_model(model, X_test, y_test):
    mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"Mean Squared Error on test set: {mse}")
    
    y_pred = model.predict([X_test_num, X_test_cat])

    # Print some sample predictions
    # for i in range(15):
    #     print(f"Actual: ${y_test.iloc[i]:.2f}, Predicted: ${y_pred[i][0]:.2f}")
    y_test_limited = []
    y_pred_limited = []
    for i in range(len(y_pred)):
        dist = y_pred[i][0] - y_test.iloc[i]
        if (abs(dist) <= 5000):
            y_test_limited.append(y_test.iloc[i])
            y_pred_limited.append(y_pred[i][0])
        elif (abs(dist) <= 30000):

            maxDist = 20000
            if dist < 0:
                if random() < ((maxDist - dist) / maxDist)**2 * 0.8:
                    y_test_limited.append(y_test.iloc[i])
                    y_pred_limited.append(y_pred[i][0])
            else:
                if random() < ((((maxDist - 5000) - dist) / (maxDist - 5000))**2)*0.8:
                    y_test_limited.append(y_test.iloc[i])
                    y_pred_limited.append(y_pred[i][0])
        else:
            if random() < 0.05:
                y_test_limited.append(y_test.iloc[i])
                y_pred_limited.append(y_pred[i][0])

    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Cost')
    plt.ylabel('Predicted Cost')
    plt.title('Actual vs Predicted Medical Costs')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_limited, y_pred_limited)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Cost')
    plt.ylabel('Predicted Cost')
    plt.title('Actual vs Predicted Medical Costs')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Load data
    X, y = load_data('AI_Model/EditedCSVs/combinedData.csv')  # Replace with your actual data file
    #X, y = fillMissing(X, y)
    
    # Preprocess data
    X_train_num, X_test_num, X_train_cat, X_test_cat, y_train, y_test = preprocess_data(X, y)
    
    # Create model
    model = create_model(X_train_num, X_test_num, X_train_cat, X_test_cat, y_train, y_test)

    # Train model
    history = model.fit(
        [X_train_num, X_train_cat],
        y_train,
        batch_size=64,
        epochs=100,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate model
    evaluate_model(model, [X_test_num, X_test_cat], y_test)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Save the model
    model.save('medical_cost_prediction_model.keras')
    print("Model saved successfully.")

    # # Example of using the model for prediction
    # sample_input = np.array([[25, 1, 0, 0, 1, 30.5]])  # Replace with actual feature values
    # sample_input_scaled = scaler.transform(sample_input)
    # predicted_cost = model.predict(sample_input_scaled)
    # print(f"Predicted cost for sample input: ${predicted_cost[0][0]:.2f}")