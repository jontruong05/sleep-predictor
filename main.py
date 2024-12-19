#!/usr/bin/env python
# coding: utf-8

# Run "streamlit run main.py" to see locally

# CS foreach Curriculum Workshop 10/24/2024: Introduction to AI/ML

# Import all relevant libraries
import pandas as pd
import numpy as np
import streamlit as st
import random
from sklearn.preprocessing import OneHotEncoder

# Title and description
st.title('Sleep Quality Predictor')
st.write("""Welcome to the Sleep Quality Predictor! This webpage is a supplemental demo 
         for CS foreach\'s Intro to AI/ML Workshop. It aims to create a simple linear 
         regression model to predict sleep quality on a scale of 1-10 based on other 
         features in the dataset.
         """)

st.header("Data Gathering and Cleaning")

st.write("""
         Below is the dataset used to predict sleep quality. Inspect the DataFrame below:
         """)

# Load the sleep data
sleep_data = pd.read_csv('Health_Sleep_Statistics.csv')
st.dataframe(sleep_data)
st.write("""Source: https://www.kaggle.com/datasets/hanaksoy/health-and-sleep-statistics""")

# Data Encoding
st.write("""
         We want to try and predict the Sleep Quality Score as found in the 
         `Sleep Quality` column using the other variables that we have.
         """)
st.write("""
         To start, let's perform One-Hot Encoding for the following columns: 
         `Gender`, `Physical Activity Level`, `Dietary Habits`, `Sleep Disorders`, 
         and `Medication Usage`, so that all variables are represented by quantities. 
         """)

# Initialize sklearn's One Hot Encoder
encoder = OneHotEncoder()

# Perform One-Hot Encoding on "Gender"
st.write("""
         Here's the One Hot Encoding for the `Gender` column:
        """)
encoded_gender = encoder.fit_transform(sleep_data[['Gender']])
gender_df = pd.DataFrame(encoded_gender.toarray(), columns=encoder.get_feature_names_out(['Gender']))
st.write(gender_df)

# Perform One-Hot Encoding on "Physical Activity Level"
st.write("""
         Here's the One Hot Encoding for the `Physical Activity Level` column:
        """)
encoded_physical_activity = encoder.fit_transform(sleep_data[['Physical Activity Level']])
physical_activity_df = pd.DataFrame(encoded_physical_activity.toarray(), columns=encoder.get_feature_names_out(['Physical Activity Level']))
st.write(physical_activity_df)

# Perform One-Hot Encoding on "Dietary Habits"
st.write("""
         Here's the One Hot Encoding for the `Dietary Habits` column:
        """)
encoded_dietary_habits = encoder.fit_transform(sleep_data[['Dietary Habits']])
dietary_habits_df = pd.DataFrame(encoded_dietary_habits.toarray(), columns=encoder.get_feature_names_out(['Dietary Habits']))
st.write(dietary_habits_df)

# Perform One-Hot Encoding on "Sleep Disorders"
st.write("""
         Here's the One Hot Encoding for the `Sleep Disorders` column:
        """)
encoded_sleep_disorders = encoder.fit_transform(sleep_data[['Sleep Disorders']])
sleep_disorders_df = pd.DataFrame(encoded_sleep_disorders.toarray(), columns=encoder.get_feature_names_out(['Sleep Disorders']))
st.write(sleep_disorders_df)

# Perform One-Hot Encoding on "Medication Usage"
st.write("""
         Here's the One Hot Encoding for the `Medication Usage` column:
        """)
encoded_medication_usage = encoder.fit_transform(sleep_data[['Medication Usage']])
medication_usage_df = pd.DataFrame(encoded_medication_usage.toarray(), columns=encoder.get_feature_names_out(['Medication Usage']))
st.write(medication_usage_df)

# Join all of the One-Hot encoded data together
st.write("""
         Now that all of our qualitative data have been quantified, we can add 
         these quantified columns and drop the original columns, since we won't 
         be needing them for predictions. We can also make each row unique by 
         setting the index of each row by `User ID`. 
         """)
encoded_sleep_data = (sleep_data
                      .join(gender_df)
                      .join(physical_activity_df)
                      .join(dietary_habits_df)
                      .join(sleep_disorders_df)
                      .join(medication_usage_df))
encoded_sleep_data = encoded_sleep_data.drop(columns=['Gender', 'Physical Activity Level', 'Dietary Habits', 'Sleep Disorders', 'Medication Usage'])
encoded_sleep_data = encoded_sleep_data.set_index('User ID')
st.write(encoded_sleep_data)

# Convert time columns to minutes
st.write("""
         The only columns that need to be converted to viable quantities are "Bedtime" and "Wake-up Time". 
         We can convert these times to minutes and create a model based on that.
         """)

st.write("""
         Here is the function we use to convert the times into minutes:
         """)
code = '''
def convert_to_minutes(time):
    time_components = time.split(':')
    minutes = int(time_components[0]) * 60 + int(time_components[1])
    return minutes
'''
st.code(code, language='python')

def convert_to_minutes(time):
    time_components = time.split(':')
    minutes = int(time_components[0]) * 60 + int(time_components[1])
    return minutes

st.write("""
         We can then apply this function create new columns from the `Bedtime` 
         and `Wake-up Time` columns. We'll also drop the original columns.
         """)

# Convert bedtimes and wake-up times to minutes
bedtime_in_minutes = encoded_sleep_data['Bedtime'].apply(convert_to_minutes)
wakeuptime_in_minutes = encoded_sleep_data['Wake-up Time'].apply(convert_to_minutes)

# Add them to the encoded sleep data DataFrame
encoded_sleep_data['Bedtime - Min'] = bedtime_in_minutes
encoded_sleep_data['Wake-up Time - Min'] = wakeuptime_in_minutes
encoded_sleep_data = encoded_sleep_data.drop(columns=['Bedtime', 'Wake-up Time'])

# Display cleaned data
st.write(encoded_sleep_data)
st.write("""
         Everything is quantified now! We can start creating the linear regression model.
         """)

st.header("Model Training and Testing")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

sleep_scores = encoded_sleep_data['Sleep Quality']
data_no_scores = encoded_sleep_data.drop(['Sleep Quality'], axis=1)

# Set the training set size
st.write("""
         Recall that when we have a dataset, we need to split it into a training
         set and a testing set. First, please set the amount of data the model 
         should train on (if unsure, try setting it anywhere between 50-90).
         """)
train_set_size = st.slider('Set the training set size (%):', 0, 100, 50, 1)
st.write(f'You are currently setting a training set size of {train_set_size}%.')

# Split into training data and testing data (conveninent that the dataset has 100 rows)
train_sleep_data, test_sleep_data, train_sleep_score, test_sleep_score = train_test_split(data_no_scores, sleep_scores, train_size=train_set_size, random_state=random.randint(1, 42))


model = LinearRegression().fit(train_sleep_data, train_sleep_score)
st.write('After training the model on the training set, the model fits the data about ' 
         + str(model.score(train_sleep_data, train_sleep_score) * 100) + '% well!')

st.write('When evaluating the model\'s performance on the testing set, it fits the testing data about ' 
         + str(model.score(test_sleep_data, test_sleep_score) * 100) + '% well!')

input_matrix = test_sleep_data.to_numpy()

st.write("""
         Let's take a look at the predicted sleep scores the model makes on the testing set:
         """)
predictions = np.dot(input_matrix, np.transpose(model.coef_)).astype(int)
st.write(predictions)

st.write("""
         Here are the actual sleep scores of the testing set:
         """)
st.write(test_sleep_score)

accuracy = np.mean(predictions == test_sleep_score)
st.write('Therefore, the model has a ' + str(accuracy * 100) + '% accuracy!')
