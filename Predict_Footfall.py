import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # Corrected the typo here
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_process_data(file_path):
    df = pd.read_csv(file_path)
    df.info()

    features = [
        "Menu_Price", "Average_Customer_Spending", "Marketing_Spend", "Promotions", "Reviews",
        "Monthly_Revenue", "Footfall_Lag_1", "Footfall_Lag_7", "Weather_Rainy", "Weather_Snowy", "Weather_Sunny",
        "Day_of_Week_Monday", "Day_of_Week_Saturday", "Day_of_Week_Sunday", "Day_of_Week_Thursday",
        "Day_of_Week_Tuesday", "Day_of_Week_Wednesday"
    ]

    X = df[features]
    y = df["Footfall_Count"]

    return X, y

def plot_predictions(y_test, rf_preds, gb_preds):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label="Actual Footfall", marker='o', linestyle='dashed')
    plt.plot(rf_preds, label="Predicted Footfall (Random Forest)", marker='s')
    plt.plot(gb_preds, label="Predicted Footfall (Gradient Boosting)", marker='^')
    plt.xlabel("Test Data Index")
    plt.ylabel("Number of Customers")
    plt.title("Actual vs. Predicted Footfall")
    plt.legend()
    plt.grid(True)
    plt.show()
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_preds = gb_model.predict(X_test)

    rf_preds = rf_preds.astype(int)
    gb_preds = gb_preds.astype(int)

    def evaluate_model(y_true, y_pred, model_name):
        print(f"\n📊 {model_name} Performance:")
        print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
        print(f"MSE: {mean_squared_error(y_true, y_pred):.2f}")
        print(f"R² Score: {r2_score(y_true, y_pred):.2f}")

    evaluate_model(y_test, rf_preds, "Random Forest")
    evaluate_model(y_test, gb_preds, "Gradient Boosting")

    results_df = pd.DataFrame({
        "Actual Footfall": y_test.values,
        "Predicted Footfall (RF)": rf_preds,
        "Predicted Footfall (GB)": gb_preds
    })
    print("\n📌 Actual vs Predicted Footfall (First 10 Entries):")
    print(results_df.head(10))

    plot_predictions(y_test, rf_preds, gb_preds)

def run_footfall_prediction():
    file_path = 'fixed_restaurant_data.csv'
    X, y = load_and_process_data(file_path)
    train_models(X, y)
