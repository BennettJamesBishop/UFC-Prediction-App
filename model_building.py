import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score
from joblib import dump
from joblib import load

# Read in matchup dataset created in data_prep file
matchup_data = pd.read_csv('matchup_data.csv')

# Test/Train Split
np.random.seed(4)  # Setting a seed so the split is the same
matchup_data['winner'] = matchup_data['winner'].astype('category')
matchup_data['r_stance'] = matchup_data['r_stance'].astype('category')
matchup_data['b_stance'] = matchup_data['b_stance'].astype('category')

X = matchup_data.drop(columns=['winner'])
y = matchup_data['winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=4)

# Preprocessing Steps (Like recipe in R version)
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['category']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the random forest model
rf = RandomForestClassifier(random_state=4)

# Create the pipeline (Like workflow)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', rf)])

# Define the parameter grid
param_grid = {
     'classifier__n_estimators': [200, 300, 400, 500, 600],
     'classifier__max_features': [1, 2, 3, 4, 5, 6],
     'classifier__min_samples_split': [10, 15, 20] }

# K-Fold Cross Validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=4)

# Perform grid search with cross-validation
# grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', verbose=3)
# grid_search.fit(X_train, y_train)

# # Select the best hyperparameters based on ROC AUC
# best_params = grid_search.best_params_
# print("Best parameters found: ", best_params)

# # Finalize the workflow with the best hyperparameters
# final_model = grid_search.best_estimator_

# # Save necessary stuff
# dump(grid_search, 'prep-RDA/tune_rf_model.joblib')
# dump(final_model, 'prep-RDA/final_rf_model.joblib')

# Load the grid search object
grid_search = load('prep-RDA/tune_rf_model.joblib')

# Load the final model
final_model = load('prep-RDA/final_rf_model.joblib')

# Evaluate model performance on the test set
# final_model.fit(X_train, y_train)
# y_pred = final_model.predict(X_test)
# y_pred_proba = final_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class (assuming 'Red')

# roc_auc = roc_auc_score(y_test, y_pred_proba)
# accuracy = accuracy_score(y_test, y_pred)
# print(f'ROC AUC on test set: {roc_auc}')
# print(f'Accuracy on test set: {accuracy}')
