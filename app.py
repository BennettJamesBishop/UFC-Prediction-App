import streamlit as st
import pandas as pd
from joblib import load
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


left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image('logo.jpg', caption=None, width=200, use_column_width=None)

# Title
with cent_co:
    st.title('UFC Fight Predictor')

# Fighter Selection in columns
col1, col2 = st.columns(2)
with col1:
    fighter1 = st.selectbox('Select Red Fighter:', fighter_stats['name'])
with col2:
    fighter2 = st.selectbox('Select Blue Fighter:', fighter_stats['name'])

# Predict Button
if st.button('Predict Winner'):
    if fighter1 != fighter2:
        matchup_data_1 = create_matchup_data(fighter1, fighter2) # Fighter 1 is Red, Fighter 2 is Blue
        matchup_data_2 = create_matchup_data(fighter2, fighter1) # Fighter 1 is Blue, Fighter 2 is Red
            
            # Use the final model's pipeline to preprocess the data and predict
        processed_data_1 = final_model.named_steps['preprocessor'].transform(matchup_data_1)
        processed_data_2 = final_model.named_steps['preprocessor'].transform(matchup_data_2)
                
                # Predict the winner
        prediction_proba_1 = final_model.named_steps['classifier'].predict_proba(processed_data_1)[0]
        prediction_proba_2 = final_model.named_steps['classifier'].predict_proba(processed_data_2)[0]
            
                #Average out two results (Eliminates bias towards blue fighters (data has 66/33 split of blue fighter winning,
                #instead of retraining model this is my work around lol))
        fighter1_prediction_proba = (prediction_proba_1[0] + prediction_proba_2[1]) / 2
        fighter2_prediction_proba = (prediction_proba_1[1] + prediction_proba_2[0]) / 2
        winner = fighter1 if fighter1_prediction_proba > fighter2_prediction_proba else fighter2
                # Change Red and Blue to fighter1 and fighter2 above for better UI
        st.subheader(f'The predicted winner is: {winner}')
        if winner == fighter1:
            st.write(f'{fighter1} has a {100 * fighter1_prediction_proba:.2f}% chance of winning')
        else:
            st.write(f'{fighter2} has a {100 * fighter2_prediction_proba:.2f}% chance of winning')
else:
    st.write('Please select two different fighters.')

# To run this Streamlit app, save it in a file (e.g., app.py) and run `streamlit run app.py` in your terminal.
