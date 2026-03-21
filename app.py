import sys
import argparse
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


CONFIG = {
    "default_csv": "house_data.csv",
    "test_size": 0.2,
    "random_state": 42 # Answer to life, universe and everything :)
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s -%(message)s"
)

logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Linear Regression analysis on housing data')
    parser.add_argument(
        "-f", "--file",
        type=str,
        default=CONFIG["default_csv"],
        help=f'Path to CSV file (default: {CONFIG["default_csv"]})'
    )
    return parser.parse_args()


def load_data(file_path):
    if not os.path.isfile(file_path):
        logger.error(f'File does not exist: {file_path}')
        sys.exit(1)

    try:
        logger.info(f'Loading data from {file_path}')
        df = pd.read_csv(file_path)

        # Validate for reqired columns
        required_columns = ["square_footage", "price_thousands"]
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column {col} not found in dataframe")
                sys.exit(1)
        return df

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")



def preprocess_data(df):
    logger.info("Preprocessing data")
    processed_df = df.copy()
    
    # Handle missing data
    if processed_df[["square_footage", "price_thousands"]].isna().any().any():
        logger.warning("Missing values found, dropping rows with missing values")
        processed_df = processed_df.dropna(subset=["square_footage", "price_thousands"])

    # Filter outliers
    for col in ["square_footage", "price_thousands"]:
        mean = processed_df[col].mean()
        std = processed_df[col].std()

        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std

        outliers = (processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)
        if outliers.any():
            logger.warning(f"Removing {outliers.sum()} outliers from {col}")
            processed_df = processed_df[~outliers]
    
    # Ensure numeric types
    processed_df["square_footage"] = pd.to_numeric(processed_df["square_footage"], errors="coerce")
    processed_df["price_thousands"] = pd.to_numeric(processed_df["price_thousands"], errors="coerce")
    processed_df = processed_df.dropna(subset=["square_footage","price_thousands"])
    
    return processed_df


def train_model (X_train, y_train):
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Train the model
    model = LinearRegression()
    model.fit(X_scaled, y_train)
    return model, scaler



def evaluate_model(model, X, y, scaler):
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    # Calculate quality metrics
    r_squared = r2_score(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    return predictions, r_squared, rmse

def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load and preprocess data
    df = load_data(args.file)
    processed_df = preprocess_data(df)
    
    # Prepare data for modeling
    X = processed_df["square_footage"].values.reshape(-1, 1) # 2D array required by sklearn
    y = processed_df["price_thousands"].values

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size= CONFIG["test_size"],
        random_state=CONFIG["random_state"]
    )

    logger.info(f"Data split: {len(X_train)} training samples, {len(X_test)} test samples")

    # Train a model
    model, scaler = train_model(X_train, y_train)
    logger.info("Model training complete")

    # Evaluate model on both trainin and test data
    train_predictions, train_r2, train_rmse = evaluate_model(model, X_train, y_train, scaler)
    test_predictions, test_r2, test_rmse = evaluate_model(model, X_test, y_test, scaler)
    
    logger.info(f"model evaluation complete. R-squared (train): {train_r2:.4f}, R-squared (test): {test_r2:.4f}")
    
    # Print results

    #Create a visualization
    
    #Predict price for houses not in our dataset

if __name__ == "__main__":
    main()