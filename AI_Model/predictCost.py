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




# Enable Metal GPU acceleration
try:
    tf.config.experimental.set_visible_devices(
        tf.config.list_physical_devices('GPU')[0], 'GPU'
    )
except IndexError:
    print("No GPU found. Using CPU.")

# Step 1: Load and prepare the data
def load_data(file_path):
    data = pd.read_csv(file_path)
    # columns_to_convert = [527,528,529,530,531,532,533,935,936,937,960,978,980,981,982,1449,1450]

    # for col in columns_to_convert:
    #     # data.iloc[:, col] = pd.to_numeric(data.iloc[:, col], errors='coerce')
    #     data.drop(data.columns[columns_to_convert], axis=1)

    data = data.replace(' ', np.nan)
    print(f"Replaced empty strings with {np.nan}")

    for column in data.columns:
        # Convert column to numeric, coercing errors to NaN
        data[column] = pd.to_numeric(data[column], errors='coerce')
        
        # Calculate median
        median_value = data[column].median()
        
        # Fill NaN values with median
        data[column].fillna(median_value, inplace=True)
    
    data = data[data['tcinpsty'].le(100000) & data['tcinpsty'].notna()]

    # Step 3: Reset the index of the filtered DataFrame (optional)
    data = data.reset_index(drop=True)


    # Verify that there are no more missing values
    missing_values_after = data.isnull().any().any()
    print("Are there any missing values?", missing_values_after)


    X = data.drop(columns=['tcinpsty', 'pnum2'], axis=1)  # Assuming 'cost' is the target column
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
    
    print("Preprocessed Data")
    #return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    return X_train, X_test, y_train, y_test


# # Step 2: Preprocess the data
# def preprocess_data(X, y):
#     # Identify numeric and categorical columns
#     numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
#     categorical_features = X.select_dtypes(include=['object']).columns

#     # Create preprocessing steps for numeric and categorical data
#     numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
#     categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
#      # Combine preprocessing steps
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numeric_transformer, numeric_features),
#             ('cat', categorical_transformer, categorical_features)
#         ])

#     # Fit the preprocessor and transform the features
#     X_preprocessed = preprocessor.fit_transform(X)

#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)
    
#     return X_train, X_test, y_train, y_test, preprocessor



# Step 3: Create the model
def create_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    print("Created the Model")
    return model



# def create_model(input_dim=1944):
#     model = Sequential([
#         # Input layer
#         Dense(2048, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.01)),
#         BatchNormalization(),
#         Dropout(0.3),
        
#         # Hidden layers
#         Dense(1024, activation='relu', kernel_regularizer=l2(0.01)),
#         BatchNormalization(),
#         Dropout(0.3),
        
#         Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
#         BatchNormalization(),
#         Dropout(0.3),
        
#         Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
#         BatchNormalization(),
#         Dropout(0.3),
        
#         Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
#         BatchNormalization(),
#         Dropout(0.3),
        
#         # Output layer
#         Dense(1)  # Single output neuron
#     ])
    
#     # Compile the model
#     model.compile(optimizer=Adam(learning_rate=0.001),
#                   loss='mean_squared_error',
#                   metrics=['mean_absolute_error'])
    
#     return model

# # Step 3: Create the model
# def create_model(input_dim):
#     model = tf.keras.Sequential([
#         Dense(512, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.01)),
#         BatchNormalization(),
#         Dropout(0.3),
#         Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
#         BatchNormalization(),
#         Dropout(0.3),
#         Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
#         BatchNormalization(),
#         Dropout(0.3),
#         Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
#         BatchNormalization(),
#         Dense(1)
#     ])
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
#     return model

# Step 4: Train the model
def train_model(model, X_train, y_train, X_test, y_test):
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    return history

# # Step 4: Train the model
# def train_model(model, X_train, y_train, X_test, y_test):
#     early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
#     reduce_lr = ReduceLROnPlateau(factor=0.2, patience=5, min_lr=0.00001)
    
#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_test, y_test),
#         epochs=200,
#         batch_size=64,
#         callbacks=[early_stopping, reduce_lr],
#         verbose=1
#     )
#     return history

# Step 5: Evaluate the model
def evaluate_model(model, X_test, y_test):
    mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"Mean Squared Error on test set: {mse}")
    
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Cost')
    plt.ylabel('Predicted Cost')
    plt.title('Actual vs Predicted Medical Costs')
    plt.show()

def evaluate_forest(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on test set: {mse}")
    
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred)
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
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Create model
    #model = create_model(X_train.shape[1])

    model = RandomForestRegressor()
    history = model.fit(X_train, y_train)

    # Train model
    #history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluate model
    evaluate_forest(model, X_test, y_test)
    
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