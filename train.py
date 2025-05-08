import pandas as pd
import numpy as np
import gdown
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import xgboost as xgb

def download_dataset():
    file_id = '1LFheHKu7kZyYJaxDcfdSp6xb2qN_X3Rq'
    url = f'https://drive.google.com/uc?id={file_id}'
    output_file = 'injuries.csv'
    gdown.download(url, output_file, quiet=False)
    return output_file

# def load_and_prepare_data(file_path):
#     # Load dataset
#     df = pd.read_csv(file_path)
#     # Placeholder for preprocessing to create merged_df_cleaned
#     # In practice, include your preprocessing steps here
#     # For now, assume merged_df_cleaned is available
#     merged_df_cleaned = df  # Replace with actual preprocessing
#     return merged_df_cleaned

# def define_features_and_target(merged_df_cleaned):
#     y = merged_df_cleaned['Injured']
#     X = merged_df_cleaned.drop(columns=[
#         'Injured', 'PLAYER_NAME', 'SEASON', 'SEASON_NUM', 'TEAM', 'INJURED ON',
#         'RETURNED', 'DAYS MISSED', 'INJURED_TYPE', 'Team', 'Notes',
#         'Out Indefinitely', 'injury_type', 'body_part', 'Date', 'year', 'month',
#         'lower body injury'
#     ])
#     return X, y

# def split_data(X, y):
#     X_temp, X_test, y_temp, y_test = train_test_split(
#         X, y, test_size=0.20, stratify=y, random_state=42
#     )
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
#     )
#     return X_train, X_val, X_test, y_train, y_val, y_test

# def define_pipeline_and_models():
#     preprocessor = Pipeline([
#         ('imputer', SimpleImputer(strategy='mean')),
#         ('scaler', StandardScaler())
#     ])
#     models = {
#         'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
#         'decision_tree': DecisionTreeClassifier(random_state=42),
#         'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
#         'xgboost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
#         'bagging': BaggingClassifier(n_estimators=50, random_state=42)
#     }
#     return preprocessor, models

# def train_models(X_train, y_train, X_val, y_val, preprocessor, models):
#     for name, clf in models.items():
#         pipeline = Pipeline([
#             ('prep', preprocessor),
#             ('clf', clf)
#         ])
#         pipeline.fit(X_train, y_train)
#         val_acc = pipeline.score(X_val, y_val)
#         print(f"{name} validation accuracy: {val_acc:.4f}")
#         joblib.dump(pipeline, f'model_{name}.pkl')

# if __name__ == "__main__":
#     # Download and load data
#     file_path = download_dataset()
#     merged_df_cleaned = load_and_prepare_data(file_path)
    
#     # Define features and target
#     X, y = define_features_and_target(merged_df_cleaned)
    
#     # Split data
#     X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
#     # Save test split
#     joblib.dump((X_test, y_test), 'test_data.pkl')
    
#     # Define pipeline and models
#     preprocessor, models = define_pipeline_and_models()
    
#     # Train models and save weights
#     train_models(X_train, y_train, X_val, y_val, preprocessor, models)

def load_and_prepare_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Clean string values like ' None'
    df = df.replace(' None', np.nan)
    
    # Convert numeric columns to proper types, handling errors
    numeric_cols = [
        'AGE', 'PLAYER_HEIGHT_INCHES', 'PLAYER_WEIGHT', 'GP', 'MIN', 'USG_PCT',
        'PACE', 'POSS', 'FGA_PG', 'DRIVES', 'DRIVE_FGA', 'DRIVE_PASSES',
        'DIST_MILES', 'AVG_SPEED', 'PULL_UP_FGA', 'PULL_UP_FG3A', 'TOUCHES',
        'FRONT_CT_TOUCHES', 'AVG_SEC_PER_TOUCH', 'AVG_DRIB_PER_TOUCH',
        'ELBOW_TOUCHES', 'POST_TOUCHES', 'PAINT_TOUCHES', 'DAYS MISSED'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create 'Injured' column: 1 if DAYS MISSED > 0, else 0
    df['Injured'] = (df['DAYS MISSED'] > 0).astype(int)
    
    return df

def define_features_and_target(merged_df_cleaned):
    # Select numeric columns only
    X = merged_df_cleaned.select_dtypes(include=['float64', 'int64']).drop(columns=['Injured', 'DAYS MISSED'], errors='ignore')
    
    # Target
    y = merged_df_cleaned['Injured']
    
    return X, y

def split_data(X, y):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def define_pipeline_and_models():
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'xgboost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'bagging': BaggingClassifier(n_estimators=50, random_state=42)
    }
    return preprocessor, models

def train_models(X_train, y_train, X_val, y_val, preprocessor, models):
    for name, clf in models.items():
        pipeline = Pipeline([
            ('prep', preprocessor),
            ('clf', clf)
        ])
        pipeline.fit(X_train, y_train)
        val_acc = pipeline.score(X_val, y_val)
        print(f"{name} validation accuracy: {val_acc:.4f}")
        joblib.dump(pipeline, f'model_{name}.pkl')

if __name__ == "__main__":
    # Download and load data
    file_path = download_dataset()
    merged_df_cleaned = load_and_prepare_data(file_path)
    
    # Define features and target
    X, y = define_features_and_target(merged_df_cleaned)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Save test split
    joblib.dump((X_test, y_test), 'test_data.pkl')
    
    # Define pipeline and models
    preprocessor, models = define_pipeline_and_models()
    
    # Train models and save weights
    train_models(X_train, y_train, X_val, y_val, preprocessor, models)