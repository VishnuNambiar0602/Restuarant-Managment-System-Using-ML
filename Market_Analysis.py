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

    # Convert timestamps to datetime
    data["order_time"] = pd.to_datetime(data["order_time"], errors='coerce')
    data["delivered_time"] = pd.to_datetime(data["delivered_time"], errors='coerce')

    # Calculate delivery time in hours
    data["delivery_time"] = (data["delivered_time"] - data["order_time"]).dt.total_seconds() / 3600

    # Drop rows with missing delivery_time
    data = data.dropna(subset=["delivery_time"])

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
    print(f"\n⏳ The order will be delivered in **{predicted_time:.2f} hours.**")

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

# Run the analysis

