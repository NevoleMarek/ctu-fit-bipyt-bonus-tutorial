import os
import time

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv


def setup_state():
    if 'first_run' not in st.session_state:
        st.session_state.first_run = True
        load_dotenv()
        st.session_state.model_service_url = os.getenv('MODEL_SERVICE_URL')

    if 'inputs' not in st.session_state:
        st.session_state.inputs = pd.DataFrame()
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []


def inference_ui():
    st.title("Inference")
    st.write("This is the inference page")
    cols = st.columns(3)
    with cols[0]:
        st.text_input('Name', key='input__name')
        st.selectbox('Class', options=[1, 2, 3], key='input__class')
        st.radio('Sex', options=['Male', 'Female'], horizontal=True, key='input__sex')
        st.number_input('Fare', key='input__fare', min_value=0.0, max_value=1000.0)
        st.button("Predict", key='button__predict')


def statistics_ui():
    st.title("Statistics")
    st.write("This is the statistics page")

    if st.session_state.inputs.empty:
        st.write("No data available")
        return

    inputs_df = st.session_state.inputs.copy()
    inputs_df['prediction'] = st.session_state.predictions
    st.dataframe(inputs_df)


def setup_ui():
    inf_tab, stat_tab = st.tabs(["Inference", "Statistics"])
    with inf_tab:
        inference_ui()
    with stat_tab:
        statistics_ui()


def infer_model():
    # Prepare the data
    data = {
        'name': st.session_state.input__name,
        'class': st.session_state.input__class,
        'sex': st.session_state.input__sex,
        'fare': st.session_state.input__fare,
    }
    endpoint = st.session_state.model_service_url + "/"

    # Call the API
    with st.spinner('Calling the API...'):
        result = requests.get(endpoint, timeout=3)

    info_message = st.empty()
    FLASH_DURATION = 1
    if result.status_code == 200:
        # Flash the success message
        info_message.success('Done!')
        time.sleep(FLASH_DURATION)
        info_message.empty()

        # Save the input and prediction
        st.session_state.inputs = pd.concat(
            [st.session_state.inputs, pd.DataFrame([data])], ignore_index=True
        )
        st.session_state.predictions.append(result.json())
    else:
        # Flash the error message
        info_message.error('Error:', result.status_code)
        time.sleep(FLASH_DURATION)
        info_message.empty()


def handle_logic():
    if st.session_state.button__predict:
        infer_model()


def main():
    setup_state()
    handle_logic()
    setup_ui()
    print(st.session_state)


if __name__ == "__main__":
    main()
