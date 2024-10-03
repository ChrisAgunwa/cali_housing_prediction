# California Housing Price Prediction

This repository contains a machine learning project focused on predicting house prices in California using the California Housing dataset. The project uses various machine learning techniques to build and fine-tune a model, which is then deployed using Streamlit to create an interactive web application.

## Dataset Overview

The California Housing dataset originates from the 1990 U.S. Census and includes data on various housing features across different districts in California. The dataset consists of 20,640 instances, with each instance representing a block group or district. It includes 8 numerical features capturing demographic and geographic information:
- Average number of rooms per dwelling
- Population
- Median income
- House age
- Latitude and longitude
- Total number of bedrooms
- Population per household
- Households

The target variable is the median house value for each district, expressed in hundreds of thousands of dollars.

This dataset is commonly used for regression tasks, with the goal of predicting the median house price based on the available features.

## Project Objective

The primary objective of this project is to build a machine learning model capable of predicting house prices in California based on the features provided in the dataset. After model development and tuning, the model is deployed using Streamlit to allow for interactive predictions.

## Features

- **Machine Learning Models**: The project implements several regression models to predict house prices. These models are evaluated based on their performance, and the best-performing model is selected for deployment.
  
- **Model Evaluation**: The models are evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²).
  
- **Deployment**: The final model is deployed using Streamlit, a framework that allows for the creation of interactive web applications with ease. Users can input various features like average income, number of rooms, and house age to predict house prices in real-time.

## Installation

To run this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/chrisagunwa/cali_housing_prediction.git
