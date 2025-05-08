import joblib
import pandas as pd
import numpy as np

def create_sample_input(X_columns):
    """Create a sample input with hardcoded realistic values."""
    sample = {col: 0.0 for col in X_columns}
    # Hardcoded realistic values based on typical NBA player stats
    defaults = {
        'AGE': 27,
        'PLAYER_HEIGHT_INCHES': 78,
        'PLAYER_WEIGHT': 220,
        'GP': 60,
        'MIN': 1800,
        'USG_PCT': 0.2,
        'PACE': 100,
        'POSS': 2000,
        'FGA_PG': 10,
        'DRIVES': 5,
        'DRIVE_FGA': 3,
        'DRIVE_PASSES': 2,
        'DIST_MILES': 150,
        'AVG_SPEED': 4.5,
        'PULL_UP_FGA': 3,
        'PULL_UP_FG3A': 1,
        'TOUCHES': 50,
        'FRONT_CT_TOUCHES': 20,
        'AVG_SEC_PER_TOUCH': 2,
        'AVG_DRIB_PER_TOUCH': 1,
        'ELBOW_TOUCHES': 5,
        'POST_TOUCHES': 3,
        'PAINT_TOUCHES': 10
    }
    for col in defaults:
        if col in sample:
            sample[col] = defaults[col]
    return pd.DataFrame([sample])

def infer_model(model_name, X_sample):
    """Load model and predict injury status."""
    try:
        pipeline = joblib.load(f'model_{model_name}.pkl')
        prediction = pipeline.predict(X_sample)
        return prediction[0]
    except FileNotFoundError:
        print(f"Error: Model file 'model_{model_name}.pkl' not found.")
        return None

if __name__ == "__main__":
    print("Welcome to the NBA Injury Prediction Tool.")
    print("This tool predicts player injury status using a hardcoded sample input.")
    
    # Load test data to get feature columns
    try:
        X_test, _ = joblib.load('test_data.pkl')
    except FileNotFoundError:
        print("Error: 'test_data.pkl' not found. Run train.py first.")
        exit()
    
    # Create sample input
    X_sample = create_sample_input(X_test.columns)
    
    # Available models
    models = ['logistic_regression', 'decision_tree', 'random_forest', 'xgboost', 'bagging']
    
    while True:
        print("\nAvailable models:", ", ".join(models))
        model_choice = input("Enter the model name or 'exit' to quit: ").strip().lower()
        
        if model_choice == 'exit':
            break
        
        if model_choice in models:
            selected_model = model_choice
            print(f"\nSelected model: {selected_model}")
            print("Using hardcoded sample input with realistic NBA player stats.")
            
            # Run inference
            prediction = infer_model(selected_model, X_sample)
            if prediction is not None:
                print(f"Prediction: {'Injured' if prediction == 1 else 'Not Injured'}")
        else:
            print("Invalid model name. Please choose from the available models.")
    
    print("Exiting program.")