import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.model_selection import  StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from joblib import dump
from joblib import load

# Load the pre-trained model and data
final_model = load('prep-RDA/final_rf_model.joblib')
fighter_stats = pd.read_csv('fighter_stats.csv')

# Function to preprocess fighter data and create a matchup dataframe
def create_matchup_data(fighter1, fighter2):
    fighter1_data = fighter_stats[fighter_stats['name'] == fighter1].iloc[0]
    fighter2_data = fighter_stats[fighter_stats['name'] == fighter2].iloc[0]
    
    matchup_data = pd.DataFrame({
        'r_stance': [fighter1_data['stance']],
        'b_stance': [fighter2_data['stance']],
        'career_wins_diff': [fighter1_data['wins'] - fighter2_data['wins']],
        'career_losses_diff': [fighter1_data['losses'] - fighter2_data['losses']],
        'age_diff': [fighter1_data['age'] - fighter2_data['age']],
        'height_diff': [fighter1_data['height'] - fighter2_data['height']],
        'weight_diff': [fighter1_data['weight'] - fighter2_data['weight']],
        'reach_diff': [fighter1_data['reach'] - fighter2_data['reach']],
        'SLpM_diff': [fighter1_data['SLpM'] - fighter2_data['SLpM']],
        'SApM_diff': [fighter1_data['SApM'] - fighter2_data['SApM']],
        'sig_str_acc_diff': [fighter1_data['sig_str_acc'] - fighter2_data['sig_str_acc']],
        'td_acc_diff': [fighter1_data['td_acc'] - fighter2_data['td_acc']],
        'str_def_diff': [fighter1_data['str_def'] - fighter2_data['str_def']],
        'td_def_diff': [fighter1_data['td_def'] - fighter2_data['td_def']],
        'sub_avg_diff': [fighter1_data['sub_avg'] - fighter2_data['sub_avg']],
        'td_avg_diff': [fighter1_data['td_avg'] - fighter2_data['td_avg']]
    })
    
    return matchup_data

# Title
st.title('UFC Fight Predictor')

# Fighter Selection
fighter_list = fighter_stats['name'].tolist()
fighter1 = st.selectbox('Select Red Fighter:', fighter_list)
fighter2 = st.selectbox('Select Blue Fighter:', fighter_list)

# Predict Button
if st.button('Predict Winner'):
    if fighter1 != fighter2:
        matchup_data = create_matchup_data(fighter1, fighter2)
        
        # Use the final model's pipeline to preprocess the data and predict
        processed_data = final_model.named_steps['preprocessor'].transform(matchup_data)
        
        # Predict the winner
        prediction_proba = final_model.named_steps['classifier'].predict_proba(processed_data)[0]
        winner = 'Red' if prediction_proba[0] > prediction_proba[1] else 'Blue'
        
        st.write(f'The predicted winner is: {winner}')
        st.write(f'Prediction probabilities - Red: {prediction_proba[0]:.2f}, Blue: {prediction_proba[1]:.2f}')
    else:
        st.write('Please select two different fighters.')

# To run this Streamlit app, save it in a file (e.g., app.py) and run `streamlit run app.py` in your terminal.
