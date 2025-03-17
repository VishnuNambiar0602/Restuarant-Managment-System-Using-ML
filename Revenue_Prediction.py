import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    start_date = pd.to_datetime("2020-01-01")  # Example start date
    df["Date"] = [start_date + timedelta(days=i) for i in range(len(df))]
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    return df

def plot_time_series(ts, title="Time Series Data"):
    plt.figure(figsize=(12,6))
    plt.plot(ts, label=title)
    plt.title(title)
    plt.legend()
    plt.show()

def decompose_time_series(ts, period=7):
    result = seasonal_decompose(ts, model='additive', period=period)
    result.plot()
    plt.show()
    return result

def train_sarima_model(train_data):
    model = SARIMAX(train_data, order=(1,1,1), seasonal_order=(1,1,1,7)).fit()
    return model

def evaluate_model(test, forecast):
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = mean_absolute_percentage_error(test, forecast) * 100
    print(f"RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

def plot_actual_vs_predicted(test, forecast):
    plt.figure(figsize=(12,6))
    plt.plot(test.index, test, label='Actual')
    plt.plot(test.index, forecast, label='Predicted', linestyle='dashed')
    plt.legend()
    plt.title("Actual vs Predicted Revenue")
    plt.show()

def forecast_future(model, ts, future_steps=30):
    future_forecast = model.predict(start=len(ts), end=len(ts) + future_steps - 1, dynamic=False)
    df_forecast = pd.DataFrame({"Date": pd.date_range(start=ts.index[-1] + timedelta(days=1), periods=future_steps), "Predicted_Revenue": future_forecast.values})
    df_forecast.to_csv("future_forecast.csv", index=False)
    print("Forecasting complete. Results saved.")
    return df_forecast

def evaluate_model(y_test, y_pred): # Changed argument names to y_test, y_pred
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    print(f"RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

def run_sales_forecasting0():
    file_path = r"fixed_restaurant_data (1).csv"
    df = load_and_prepare_data(file_path)
    target_col = "Monthly_Revenue"
    ts = df[target_col]

    plot_time_series(ts, title="Monthly Revenue Over Time")
    decompose_time_series(ts, period=7)

    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]

    sarima_model = train_sarima_model(train)
    forecast = sarima_model.predict(start=len(train), end=len(ts)-1, dynamic=False)

    evaluate_model(test, forecast) # Calling evaluate_model with test as y_test and forecast as y_pred
    plot_actual_vs_predicted(test, forecast)

    forecast_future(sarima_model, ts, future_steps=30)
def run_sales_forecasting():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Load dataset
    file_path = r"fixed_restaurant_data.csv"
    df = pd.read_csv(file_path)

    # Use a more recent start date to ensure 2025 predictions
    start_date = pd.to_datetime("2024-01-01")
    df["Date"] = [start_date + pd.Timedelta(days=i) for i in range(len(df))]

    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    ts = df["Monthly_Revenue"]

    # Train SARIMA model
    sarima_model = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,7)).fit()

    # Get user input
    future_steps = int(input("\n📅 Enter number of days to forecast: "))

    # Predict future revenue
    future_forecast = sarima_model.predict(start=len(ts), end=len(ts) + future_steps - 1, dynamic=False)

    # Ensure predictions align with 2025
    df_forecast = pd.DataFrame({"Date": pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=future_steps),
                                "Predicted_Revenue": future_forecast.values})

    print("\n📊 Predicted Revenue for Next Days:\n", df_forecast)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocess data
def process_data(data):
    # Drop unnecessary columns
    data = data.drop(columns=['reassignment_method', 'reassignment_reason', 'cancelled_time'], errors='ignore')

    # Fill missing numerical values with median
    num_cols = ['alloted_orders', 'delivered_orders', 'undelivered_orders', 
                'lifetime_order_count', 'reassigned_order', 'session_time']
    
    for col in num_cols:
        data[col] = data[col].fillna(data[col].median())

    # Convert timestamps to datetime with correct format
    data["order_time"] = pd.to_datetime(data["order_time"], format="%Y-%m-%d %H:%M:%S", errors='coerce')
    data["delivered_time"] = pd.to_datetime(data["delivered_time"], format="%Y-%m-%d %H:%M:%S", errors='coerce')

    # Remove rows where date parsing failed
    data = data.dropna(subset=["order_time", "delivered_time"])

    # Calculate delivery time in hours
    data["delivery_time"] = (data["delivered_time"] - data["order_time"]).dt.total_seconds() / 3600

    # Remove unrealistic values (e.g., future dates)
    data = data[(data["order_time"] < pd.Timestamp.now()) & (data["delivered_time"] < pd.Timestamp.now())]

    return data

# Train model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae

# Predict delivery time based on user input
def predict_delivery_time(model, data):
    print("\n📍 Enter Delivery Details:\n")
    first_mile = float(input("🚴 Enter First Mile Distance (km): "))
    last_mile = float(input("🏠 Enter Last Mile Distance (km): "))

    # Define features used in the model
    features = ['first_mile_distance', 'last_mile_distance', 'alloted_orders', 
                'delivered_orders', 'undelivered_orders', 'lifetime_order_count', 
                'reassigned_order', 'order_to_allot', 'allot_to_accept', 'accept_to_pickup', 'pickup_to_delivery']

    # Ensure selected features exist in the dataset
    features = [f for f in features if f in data.columns]

    # Prepare input data with user values and median values for missing ones
    input_data = pd.DataFrame([{
        'first_mile_distance': first_mile,
        'last_mile_distance': last_mile,
        **{col: data[col].median() for col in features if col not in ['first_mile_distance', 'last_mile_distance']}
    }])

    # Predict delivery time
    predicted_time = model.predict(input_data)[0]
    print(f"\n⏳ Prediction: The order will be delivered in **{predicted_time:.2f} hours.**")

# Main execution
def run_market_analysis():
    file_path = 'Rider-Info.csv'
    data = load_data(file_path)
    data = process_data(data)

    target = 'delivery_time'
    features = ['first_mile_distance', 'last_mile_distance', 'alloted_orders', 'delivered_orders',
                'undelivered_orders', 'lifetime_order_count', 'reassigned_order',
                'order_to_allot', 'allot_to_accept', 'accept_to_pickup', 'pickup_to_delivery']

    # Ensure features exist in the dataset
    features = [f for f in features if f in data.columns]

    X = data[features]
    y = data[target]

    # Handle missing or invalid target values
    X, y = X[~y.isna()], y.dropna()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    mae = evaluate_model(model, X_test, y_test)

    print("\n🎯 Model Performance:")
    print(f"Mean Absolute Error: {mae:.2f} hours")

    # Get user input and predict delivery time
    predict_delivery_time(model, data)

