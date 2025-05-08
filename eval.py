import joblib
from sklearn.metrics import classification_report, confusion_matrix

def load_test_data():
    X_test, y_test = joblib.load('test_data.pkl')
    return X_test, y_test

def evaluate_models(X_test, y_test):
    models = ['logistic_regression', 'decision_tree', 'random_forest', 'xgboost', 'bagging']
    for name in models:
        pipeline = joblib.load(f'model_{name}.pkl')
        y_pred = pipeline.predict(X_test)
        print(f"\n== Evaluation: {name} ==")
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    # Load test data
    X_test, y_test = load_test_data()
    
    # Evaluate models
    evaluate_models(X_test, y_test)