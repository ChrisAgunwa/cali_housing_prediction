import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load the California Housing dataset
california_housing = fetch_california_housing()

# Select only the relevant features
X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)[['MedInc', 'HouseAge', 'AveRooms']]
y = pd.Series(california_housing.target, name='MedHouseVal')

housing = X

# Set up Streamlit page
st.set_page_config(page_title="California Housing Price Predictor", page_icon="🏠", layout="wide", initial_sidebar_state="expanded")

# Page title and description
st.title("🏠 California Housing Price Predictor")
st.markdown("""
Use this tool to predict the median house value in California based on selected features from the 1990 U.S. Census. Adjust the sliders in the sidebar to simulate different scenarios and view the predicted housing price.
""")
st.write('---')

# Sidebar for user inputs
with st.sidebar:
    st.header("🏠 Set Housing Features")
    
    # Collect user input with sliders
    user_input = {
        'MedInc': st.slider('Median Income (in $10,000)', housing['MedInc'].min(), housing['MedInc'].max(), housing['MedInc'].median()),
        'HouseAge': st.slider('House Age (in years)', housing['HouseAge'].min(), housing['HouseAge'].max(), housing['HouseAge'].median()),
        'AveRooms': st.slider('Average Rooms per Dwelling', housing['AveRooms'].min(), housing['AveRooms'].max(), housing['AveRooms'].median())
    }

# Convert user input to DataFrame
user_input_df = pd.DataFrame([user_input])

# Feature scaling
scaler = StandardScaler()

# Fit the scaler to the training data and transform both X and the user input
X_scaled = scaler.fit_transform(X)
user_input_scaled = scaler.transform(user_input_df)

# Train the model on the scaled data
model = LinearRegression()
model.fit(X_scaled, y)

# Prediction
prediction = model.predict(user_input_scaled)[0]
prediction = max(prediction, 0)  # Ensure prediction is non-negative

# Display prediction with a larger metric and some formatting
st.markdown('## 💡 Predicted Median House Value')
st.metric(label="Estimated Price", value=f"${prediction * 1000:,.0f}", delta=None)

# Expandable section for feature distributions
with st.expander("🔍 Feature Distributions"):
    st.subheader("Feature Distributions")
    
    # Set up a grid for 3 plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.histplot(housing['MedInc'], bins=30, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Median Income Distribution')
    axes[0].set_xlabel('Median Income ($10,000)')
    
    sns.histplot(housing['HouseAge'], bins=30, kde=True, ax=axes[1], color='lightgreen')
    axes[1].set_title('House Age Distribution')
    axes[1].set_xlabel('House Age (years)')
    
    sns.histplot(housing['AveRooms'], bins=30, kde=True, ax=axes[2], color='lightcoral')
    axes[2].set_title('Average Rooms Distribution')
    axes[2].set_xlabel('Average Rooms')
    
    st.pyplot(fig)

# Expandable section for pairplot visualization
with st.expander("📊 Pairplot of Selected Features"):
    st.subheader("Pairplot of Features")
    
    # Generate pairplot (reduced features for simplicity)
    pairplot_fig = sns.pairplot(X)
    
    st.pyplot(pairplot_fig)


# Generate the pairplot
st.subheader("Pairplot of Scaled Features")
fig = sns.pairplot(X)
st.pyplot(fig)




"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load the California Housing dataset
california_housing = fetch_california_housing()

# Select only the relevant features
X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)[['MedInc', 'HouseAge', 'AveRooms']]
y = pd.Series(california_housing.target, name='MedHouseVal')

housing = X

# Set up Streamlit page
st.set_page_config(page_title="Housing Predictor", page_icon="🏠", layout="wide", initial_sidebar_state="expanded")
st.title("California House Price Prediction")
st.write('Use the sliders on the left to make the housing prediction.')
st.write('---')

# Sidebar for user inputs
with st.sidebar:
    st.title("Select Your Preference")
    # Collect user input
    user_input = {
        'MedInc': st.slider('Median Income', housing['MedInc'].min(), housing['MedInc'].max(), housing['MedInc'].median()),
        'HouseAge': st.slider('Housing Median Age', housing['HouseAge'].min(), housing['HouseAge'].max(), housing['HouseAge'].median()),
        'AveRooms': st.slider('Average Rooms', housing['AveRooms'].min(), housing['AveRooms'].max(), housing['AveRooms'].median()),
    }

# Convert user input to DataFrame
user_input_df = pd.DataFrame([user_input])

# Feature scaling
scaler = StandardScaler()

# Fit the scaler to the training data and transform both X and the user input
X_scaled = scaler.fit_transform(X)
user_input_scaled = scaler.transform(user_input_df)

# Train the model on the scaled data
model = LinearRegression()
model.fit(X_scaled, y)


# Input parameters and prediction
st.markdown('### Prediction')

# Predict the price using the scaled user input
prediction = model.predict(user_input_scaled)[0]

# Ensure the prediction is non-negative
prediction = max(prediction, 0)

st.metric(label="Predicted Median House Value", value=f"${prediction * 1000:,.0f}")

with st.expander("Feature Distributions"):
    st.subheader("Feature Distributions")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(housing['MedInc'], bins=30, kde=True, ax=axes[0])
    axes[0].set_title('Median Income Distribution')
    sns.histplot(housing['HouseAge'], bins=30, kde=True, ax=axes[1])
    axes[1].set_title('House Age Distribution')
    sns.histplot(housing['AveRooms'], bins=30, kde=True, ax=axes[2])
    axes[2].set_title('Average Rooms Distribution')
    st.pyplot(fig)"""




