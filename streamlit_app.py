# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 2023 16:45:56

@author: Lakshmi Muthukumar

"""
# streamlit library
import streamlit as st
import base64
from streamlit import components 
import time

# audio libraries
import librosa
import IPython
import ffmpeg

# transform .wav into .csv
import csv
import os
import numpy as np
import pandas as pd

# preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# model
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential



def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('background.jpg') 


st.write("""
# Baby Cry Monitor App
This app predicts the nature of **Baby Cry** from audio clip.
Data used for building the model here, is obtained from the [**donateacry-corpus**](https://github.com/gveres/donateacry-corpus) by Gabor Veres.
""")


st.write("""
#### \t Upload infant cry audio clip here...
""")


def from_wav_to_csv(sound_saved):

    header = "filename length chroma_stft_mean chroma_stft_var rms_mean rms_var spectral_centroid_mean spectral_centroid_var spectral_bandwidth_mean \
            spectral_bandwidth_var rolloff_mean rolloff_var zero_crossing_rate_mean zero_crossing_rate_var harmony_mean harmony_var perceptr_mean \
            perceptr_var tempo mfcc1_mean mfcc1_var mfcc2_mean mfcc2_var mfcc3_mean mfcc3_var mfcc4_mean mfcc4_var label".split()

    file = open(f'data_test.csv', 'w', newline = '')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)      
    # calculate the value of the librosa parameters
    
    y, sr = librosa.load(sound_saved, mono = True, duration = 30)
    chroma_stft = librosa.feature.chroma_stft(y = y, sr = sr)
    rmse = librosa.feature.rms(y = y)
    spec_cent = librosa.feature.spectral_centroid(y = y, sr = sr)
    spec_bw = librosa.feature.spectral_bandwidth(y = y, sr = sr)
    rolloff = librosa.feature.spectral_rolloff(y = y, sr = sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y = y, sr = sr)
    to_append = f'{sound_saved} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    # fill in the csv file
    file = open(f'data_test.csv', 'a', newline = '')
    with file:
        writer = csv.writer(file)
        writer.writerow(to_append.split())
    # create test dataframe
    df_test = pd.read_csv(f'data_test.csv')
    # each time you add a sound, a line is added to the test.csv file
    # if you want to display the whole dataframe, you can deselect the following line
    #st.write(df_test)
    return df_test



def classification(dataframe):
    # create a dataframe with the csv file of the data used for training and validation
    df = pd.read_csv('data.csv')
    # OUTPUT: labels => last column
    labels_list = df.iloc[:,-1]
    # encode the labels (0 => 44)
    converter = LabelEncoder()
    y = converter.fit_transform(labels_list)
    # INPUTS: all other columns are inputs except the filename
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(df.iloc[:, 1:27]))
    X_test = scaler.transform(np.array(dataframe.iloc[:, 1:27]))
    # load the pretrained model
    model = tf.keras.models.load_model('saved_model_clean_data/my_model')
    # generate predictions for test samples
    predictions = model.predict(X_test)
    print(predictions[0])
    baby_cry_tags = "bellyPain burping coldHot discomfort dontKnow hungry lonely scared tired".split()
    for i in range(len(predictions[0])):
        print(baby_cry_tags[i], '\t==>\t', predictions[0][i]*100, '%')
    # generate argmax for predictions
    classes = np.argmax(predictions, axis = 1)
    result = converter.inverse_transform(classes)
    print("classes", classes)
    print("result", result)
    return result

def convert_to_wav(input_file):
    extn = input_file[-3:]
    if extn.lower() == 'wav':
        return input_file
    name_without_extn = input_file[:-4]
    output_file = name_without_extn + ".wav"
    stream = ffmpeg.input(input_file)
    stream = ffmpeg.output(stream, output_file)
    ffmpeg.run(stream, overwrite_output=True)
    return output_file
    
#Add file uploader to allow users to upload photos
uploaded_file = st.file_uploader("", type=['wav','caf','3gp'])
if uploaded_file is not None:
    #print (uploaded_file)
    # Save uploaded file to disk.
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as f:
        f.write(bytes_data)
    uploaded_file = uploaded_file.name

    # Convert to wav if needed.
    wav_file = convert_to_wav(uploaded_file)

    data, sr = librosa.load(uploaded_file)
    print(type(data), type(sr))    
    print(len(data), sr)
    st.write("""
        ###   **Play  Audio**
    """)
    html = IPython.display.Audio(data, rate = sr)._repr_html_()
    components.v1.html(html)

    with st.form('predict_form'):
        submitted = st.form_submit_button('Predict')
        if submitted:
            with st.spinner('Wait for it...'):
                df_test = from_wav_to_csv(uploaded_file)
                result = classification(df_test)
                output = ''
                if result == 'hungry' or result == 'lonely' or result == 'scared' or result == 'tired':
                    output = f'Baby is {result[0]}!'
                elif result == 'bellyPain':
                    output = 'Baby has belly pain!'
                elif result == 'burping':
                    output = 'Baby needs burping!'
                elif result == 'coldHot':
                    output = 'Baby feels cold/hot!'
                elif result == 'discomfort':
                    output = 'Baby feels a general discomfort!'
                elif result == 'dontKnow':
                    output = 'Baby cry ==> unpredictable!'
                else:
                    output = 'Baby cry ==> unknown!'
                st.success('Done!')
                st.markdown(f'<p style="font-size: 20px;"><b>{output}</b></p>', unsafe_allow_html=True)
                #st.balloons()
    

#Add a header and expander in side bar
original_title = '<p style="color:Blue; font-size: 20px;"><b>Audio Prediction</b></p>'
st.sidebar.markdown(original_title, unsafe_allow_html=True)
with st.sidebar.expander("**About the App**"):
     st.markdown("""
        <span style='color:green'> Use this simple app to know why your infant is crying from this AI driven tool.
        Babies cry for a variety of reasons like hungry, needs burping, general discomfort, tired, belly pain, feeling hot/cold, scared, lonely,etc.
        This app was created for product demo. Hope you enjoy!</span>
     """, unsafe_allow_html=True)


#Add a feedback section in the sidebar
st.sidebar.title(' ') #Used to create some space between the filter widget and the comments section
st.sidebar.markdown(' ') #Used to create some space between the filter widget and the comments section
st.sidebar.subheader('Please help us improve!')
with st.sidebar.form(key='columns_in_form',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
    rating=st.slider("Please rate the app", min_value=1, max_value=5, value=3,help='Drag the slider to rate the app. This is a 1-5 rating scale where 5 is the highest rating')
    text=st.text_input(label='Please leave your feedback here')
    submitted = st.form_submit_button('Submit')
    if submitted:
      st.write('Thanks for your feedback!')
      st.markdown('Your Rating:')
      st.markdown(rating)
      st.markdown('Your Feedback:')
      st.markdown(text)
