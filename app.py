import streamlit as st
import pandas as pd
from joblib import load
import streamlit as st
import requests
import os

# Load the pre-trained model and fighter data
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

# Access the environment variable
OPENAI_API_KEY = db_username = st.secrets["OPENAI_API_KEY"]

# Function to fetch explanation from OpenAI API
def fetch_openai_explanation(fighter1, fighter2, fighter1_data, fighter2_data, prediction):
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {os.getenv("OPENAI_API_KEY")}'
    }
    content = (
        f"Fighter 1: {fighter1}\n"
        f"Fighter 2: {fighter2}\n"
        f"Fighter 1 Data: {fighter1_data}\n"
        f"Fighter 2 Data: {fighter2_data}\n"
        f"Prediction: {prediction}\n"
    )
    payload = {
        'model': 'gpt-4o-mini',
        'messages': [
            {
                'role': 'system',
                'content': (
                    "You are an expert UFC analyst. You will be given the names of two fighters, their respective stats, and the predicted winner of their fight. "
                    "In about 150 words, your task is to explain the reasoning behind the given prediction based on any five of the provided stats of both fighters. "
                    "Mention at least five statistics, specifically, the difference in each fighter's respective statistics and how that may influence the fight. "
                    "Here are the features used in order of their importance to the model, the abbreviation of the feature: "
                    "1. average strikes landed per minute (SLpM), "
                    "2. career wins (wins), "
                    "3. takedown defense (td_def), "
                    "4. average strikes absorbed per minute (SApM_diff), "
                    "5. average significant strikes accuracy (sig_str_acc), "
                    "6. takedown Average (td_avg), "
                    "7. Age (age): 0.071294, "
                    "8. Career Losses (losses), "
                    "9. Striking Defense (str_def), "
                    "10. Takedown Accuracy (td_acc), "
                    "11. Submission Average (sub_avg), "
                    "12. Reach (reach), "
                    "13. Weight (weight), "
                    "14. Height (height), "
                    "If the two fighter's weights are very different, mention that this model does not take weight into account. "
                    "Do not make your own prediction; just explain the provided one. "
                    "Make sure to end your summary at the end of a sentence. "
                    
            )
            },
            {
                'role': 'user',
                'content': content
            }
        ],
        'max_tokens': 300,
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data['choices'][0]['message']['content']

#Comment back in to have cage as background
# background_image = """
# <style>
# [data-testid="stAppViewContainer"] > .main {
#     background-image: url('https://ufcprediction.s3.us-east-2.amazonaws.com/ufc-octagon.png');
#     background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
#     background-position: center;  
#     background-repeat: no-repeat;
# }
# </style>
# """

# st.markdown(background_image, unsafe_allow_html=True)

left_co, cent_co,last_co = st.columns(3)
with cent_co:
    st.image('logo.jpg', caption=None, width=200, use_column_width=None)

# Title
with cent_co:
    st.title('UFC Fight Predictor')

# Fighter Selection in columns
col1, col2 = st.columns(2)
with col1:
    fighter1 = st.selectbox(':red[Select Red Fighter:]', fighter_stats['name'])
with col2:
    fighter2 = st.selectbox(':blue[Select Blue Fighter:]', fighter_stats['name'])

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

        #Get individual fighter data again for GPT explanation
        fighter1_data = fighter_stats[fighter_stats['name'] == fighter1].iloc[0]
        fighter2_data = fighter_stats[fighter_stats['name'] == fighter2].iloc[0]
        if winner == fighter1:
            st.subheader(f'The predicted winner is: :red[{winner}]')
            st.write(f'{fighter1} has a {100 * fighter1_prediction_proba:.2f}% chance of winning')
            winner_for_gpt = fighter1
            
        else:
            st.subheader(f'The predicted winner is: :blue[{winner}]')
            st.write(f'{fighter2} has a {100 * fighter2_prediction_proba:.2f}% chance of winning')
            winner_for_gpt = fighter2
           
         # Fetch and display explanation with a spinner
        with st.spinner('Generating AI explanation...'):
            explanation = fetch_openai_explanation(fighter1, fighter2, fighter1_data, fighter2_data, winner_for_gpt)
        st.markdown(explanation)
    else:
        st.write('Please select two different fighters.')


# To run this Streamlit app, save it in a file (e.g., app.py) and run `streamlit run app.py` in your terminal.
