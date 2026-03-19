
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model and preprocessing objects
@st.cache_resource
def load_resources():
    loaded_model = pickle.load(open('trained_model.sav', 'rb'))
    std_scaler = pickle.load(open('std_scaler.pkl', 'rb'))
    minmax_scaler = pickle.load(open('minmax_scaler.pkl', 'rb'))
    frequency_map = pickle.load(open('freq_map.pkl', 'rb'))
    x_columns = pickle.load(open('columns.pkl', 'rb'))
    return loaded_model, std_scaler, minmax_scaler, frequency_map, x_columns

xgb_model, std_scaler, min_max_scaler, frequency_map, x_columns = load_resources()

st.title('Remaining Useful Life (RUL) Prediction App')
st.write('Enter the machine parameters to predict its Remaining Useful Life.')

# Create input fields for the user
# Assuming `x_columns` contains the correct order and names of features
input_data = {}

# Machine Type input for frequency encoding
machine_types = list(frequency_map.keys())
selected_machine_type = st.selectbox('Machine Type', machine_types)

# Numerical inputs
for col in x_columns:
    if col == 'Machine_Type_Frequency':
        input_data[col] = frequency_map.get(selected_machine_type, 0) # Apply frequency encoding
    elif col == 'AI_Supervision':
        input_data[col] = st.selectbox(col, [0, 1]) # Assuming AI_Supervision is 0 or 1
    else:
        # Determine a reasonable default and step based on column name or prior knowledge
        default_value = 0.0 # Placeholder, ideally use mean/median/etc. from training data
        step_value = 0.1
        if 'Year' in col or 'Hours' in col or 'Days_Ago' in col or 'Count' in col or 'Errors' in col:
            default_value = 0 # Integer-like defaults
            step_value = 1
        elif 'pct' in col:
            default_value = 50.0
            step_value = 1.0

        input_data[col] = st.number_input(f'{col}', value=float(default_value), step=float(step_value))


if st.button('Predict RUL'):
    # Convert input data to a DataFrame
    new_df_input = pd.DataFrame([input_data])

    # Ensure the column order matches the training data
    new_df_input = new_df_input[x_columns]

    # Apply scaling transformations
    std_cols = ['Temperature_C', 'Vibration_mms', 'Sound_dB', 'Power_Consumption_kW']
    minmax_cols = ['Installation_Year', 'Operational_Hours', 'Oil_Level_pct', 'Coolant_Level_pct',
                   'Last_Maintenance_Days_Ago', 'Maintenance_History_Count', 'Failure_History_Count',
                   'AI_Supervision', 'Error_Codes_Last_30_Days']

    new_df_input[std_cols] = std_scaler.transform(new_df_input[std_cols])
    new_df_input[minmax_cols] = min_max_scaler.transform(new_df_input[minmax_cols])

    # Make prediction
    predicted_rul = xgb_model.predict(new_df_input)

    st.success(f'Predicted Remaining Useful Life: {predicted_rul[0]:.2f} days')

