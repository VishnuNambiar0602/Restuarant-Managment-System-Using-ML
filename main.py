def main():
    while True:
        print("\n📊 Welcome to AI-Powered Restaurant Analytics System!")
        print("Choose a model to run:")
        print("1️⃣ Restaurant Sales Forecasting (SARIMA)")
        print("2️⃣ Customer Footfall Prediction (Random Forest & GB)")
        print("3️⃣ Competitor & Market Analysis (Swiggy Dataset)")
        print("4️⃣ Exit\n")

        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            print("\n🔍 Running Restaurant Sales Forecasting Model...")
            run_sales_forecasting()

        elif choice == "2":
            print("\n📈 Running Customer Footfall Prediction Model...")
            run_footfall_prediction()

        elif choice == "3":
            print("\n📊 Running Competitor & Market Analysis Model...")
            run_market_analysis()



        elif choice == "4":
            print("\n✅ Exiting the system. Have a great day!")
            break

        else:
            print("❌ Invalid choice. Please enter a number between 1 and 4.")

# Run the main function
main()
