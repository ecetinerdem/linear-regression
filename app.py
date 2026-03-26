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
    "random_state": 42, # Answer to life, universe and everything :)
    "figure_size": (10, 6),
    "point_color": "blue",
    "line_color": "red",
    "grid_alpha": 0.3,
    "output_image": "housing_regression.png",
    "line_width": 2
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
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help= "Do not display plot (still saves to file)"
    )

    parser.add_argument(
        "-predict", "--predict-sqft",
        type=float,
        help="Predict the price for a house with the given square footage"
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


def print_results(X_train, y_train, X_test, y_test, train_predictions, test_predictions, model, scaler):
    slope = model.coef_[0] / scaler.scale_[0]
    intercept = model.intercept_ - (model.coef_[0] * scaler.mean_[0] / scaler.scale_[0])


    r_squared_train = r2_score(y_train, train_predictions)
    r_squared_test = r2_score(y_test, test_predictions)
    rmse_train = np.sqrt(mean_squared_error(y_train, train_predictions))
    rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions))

    print(f"\nLinear Regression Formula: {slope:.4f} x Square Footage + {intercept:.4f}")
    print(f"R-squared (training): {r_squared_train:.4f}")
    print(f"R-squared (test): {r_squared_test:.4f}")
    print(f"RMSE (training): {rmse_train:.4f}")
    print(f"RMSE (test): {rmse_test:.4f}")

    train_df = pd.DataFrame({
        "Square Footage": X_train.flatten(),
        "Actual Price ($K)": y_train,
        "Predicted Price ($K)": np.round(train_predictions, 2) 
    })

    test_df = pd.DataFrame({
        "Square Footage": X_test.flatten(),
        "Actual Price ($K)": y_test,
        "Predicted Price ($K)": np.round(test_predictions, 2) 
    })

    print("\nTraining Prediction Sample (first 5 rows)")
    print(train_df.head().to_string(index=False))
   
    print("\nTest Prediction Sample (first 5 rows)")
    print(test_df.head().to_string(index=False))


def create_visualization(X_train, y_train, X_test, y_test, train_predictions, test_predictions, model, scaler, output_file, show_plot=True):
    plt.figure(figsize=CONFIG["figure_size"])
    # Plot training data
    plt.scatter(X_train, y_train, color=CONFIG["point_color"], alpha=0.7, label="Training data")
    plt.scatter(X_test, y_test, color="green", alpha=0.7, label= "Test data")

    x_range = np.linspace(
        min(X_train.min(), X_test.min()),
        max(X_train.max(), X_test.max()),
        100 # 100 points for smooth line
    ).reshape(-1, 1)

    # Scale the range and predict corresponding y-values
    X_range_scaled = scaler.transform(x_range)
    y_range_predictions = model.predict(X_range_scaled)


    # Plot the regression line
    plt.plot(x_range, y_range_predictions, color=CONFIG["line_color"], linewidth=CONFIG["line_width"], label="Regression line")

    # Add labels and title
    plt.xlabel("Square Footage")
    plt.ylabel("Price (thousands $)")
    plt.title("Liner Regression: Housing Price vs Square Footage")
    plt.legend()
    plt.grid(True, alpha=CONFIG["grid_alpha"])

    # Calculate and display model parameter
    slope = model.coef_[0] / scaler.scale_[0]
    intercept = model.intercept_ - (model.coef_[0] * scaler.mean_[0] / scaler.scale_[0])

    r_squared_train = r2_score(y_train, train_predictions)
    r_squared_test = r2_score(y_test, test_predictions)

    # Format text to display on plot
    formula_text = f"Price = {slope:.4f} x Square Footage + {intercept:.4f}"
    r2_train_text = f"R2 (train): {r_squared_train:.4f}"
    r2_test_text = f"R2 (test): {r_squared_test:.4f}"

    plt.figtext(0.15, 0.85, formula_text, fontsize=12)
    plt.figtext(0.15, 0.82, r2_train_text, fontsize=12)
    plt.figtext(0.15, 0.79, r2_test_text, fontsize=12)

    plt.savefig(output_file)
    logger.info(f"Plot saved as {output_file}")

    if show_plot:
        plt.show()

    plt.close()



def predict_price(model, scaler, square_footage):
    sqft_array = np.array([[square_footage]])
    sqft_scaled = scaler.transform(sqft_array) 
    predicted_price = model.predict(sqft_scaled)

    return predicted_price[0]

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
    print_results(X_train, y_train, X_test, y_test, train_predictions , test_predictions, model, scaler)

    #Create a visualization
    create_visualization(X_train, y_train, X_test, y_test, train_predictions, test_predictions, model, scaler, CONFIG["output_image"], not args.no_plot)
    #Predict price for houses not in our dataset
    if args.predict_sqft is not None:
        sqft_to_predict = args.predict_sqft
        logger.info(f"predicting price for a house with {sqft_to_predict} square footage...")
        predicted_price = predict_price(model, scaler, sqft_to_predict)
        print(f"\nPredicted price for a house with {sqft_to_predict} square footage: ${predicted_price:.2f} thousands")
        print(f"This is equivalent approximetly ${predicted_price * 1000: .2f}")

if __name__ == "__main__":
    main()