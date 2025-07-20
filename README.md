# House Price Prediction

This repository contains a Jupyter Notebook and Streamlit app for predicting house prices using machine learning techniques. The project involves data loading, exploratory data analysis (EDA), preprocessing, and model training to predict house prices based on various features.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Streamlit App Summary](#streamlit-app-summary)
- [Model Deployment](#model-deployment)
- [Running Streamlit App](#running-streamlit-app)


## Project Overview
The goal of this project is to predict house prices using:
1. A Jupyter Notebook with complete analysis
2. An interactive Streamlit web application

The project covers:
- Data loading and initial exploration
- Data cleaning and handling missing values
- Exploratory data analysis (EDA) with visualizations
- Feature engineering and preprocessing
- Training and evaluating multiple machine learning models
- Deploying the model via Streamlit

## Dataset
The dataset used in this project is stored in `data.csv` and contains 81 features for 1460 houses. Key features include:
- `SalePrice`: The target variable (price of the house)
- `LotArea`: Lot size in square feet
- `OverallQual`: Overall material and finish quality
- `BedroomAbvGr`: Number of bedrooms above ground

## Dependencies
To run this notebook, you need the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Streamlit App Summary
The Streamlit app provides a user-friendly interface to:
- Upload new housing data for predictions
- Adjust key features using interactive sliders
- View predicted house prices in real-time
- See feature importance visualizations
- Download prediction results

**Key Features:**
- Interactive input controls
- Real-time predictions
- Model performance metrics
- Responsive design


## Model Deployment
The model is deployed using Streamlit Cloud

**Deployment Process:**
- Pushed code to GitHub repository
- Created Streamlit Cloud account
- Connected GitHub repository
- Specified main file as app/predictionapp.py

**Deployed App Live URL:**
House Price Predictor
```bash
https://houseprice-prediction-nesj6nemauux3h6zqdghfd.streamlit.app/
```

## Running Streamlit App
1. Ensure you have all dependencies installed:
```bash
pip install streamlit pandas numpy scikit-learn
streamlit run predictionapp.py
```
