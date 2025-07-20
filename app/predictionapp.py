import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler

st.set_page_config(page_title="Simple House Price Predictor", layout="wide")
st.title("🏠 Simple House Price Predictor")

@st.cache_data
def load_data():
    return pd.read_csv("NotebooksFiles/data.csv")

data = load_data()

def preprocess_data(df):
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    
    features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    X = df[features]
    y = df['SalePrice']
    return X, y

@st.cache_resource
def train_model():
    X, y = preprocess_data(data)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = train_model()

st.sidebar.header("Enter House Details")

def user_input():
    qual = st.sidebar.slider('Overall Quality (1-10 scale)', 1, 10, 5)
    area = st.sidebar.number_input('Living Area (sq ft)', min_value=500, max_value=5000, value=1500)
    garage = st.sidebar.slider('Garage Size (car capacity)', 0, 4, 2)
    basement = st.sidebar.number_input('Basement Area (sq ft)', min_value=0, max_value=3000, value=1000)
    baths = st.sidebar.slider('Number of Full Bathrooms', 0, 4, 2)
    year = st.sidebar.slider('Year Built', 1870, 2020, 2000)
    
    return pd.DataFrame({
        'OverallQual': [qual],
        'GrLivArea': [area],
        'GarageCars': [garage],
        'TotalBsmtSF': [basement],
        'FullBath': [baths],
        'YearBuilt': [year]
    })

user_data = user_input()

st.subheader("Your House Details")
st.write(user_data)

def predict(input_data):
    scaled_input = scaler.transform(input_data)
    return model.predict(scaled_input)[0]

if st.button(' Predict Price'):
    price = predict(user_data)
    st.subheader(f"Predicted Price: ${price:,.2f}")

st.subheader("Key Insights")

st.write("### House Price Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(data['SalePrice'], kde=True, color='skyblue', ax=ax1)
ax1.set_xlabel('Price ($)')
ax1.set_ylabel('Number of Houses')
st.pyplot(fig1)

st.markdown("""
**What this tells us:**
- Shows how house prices are distributed in our dataset
- The curve shows most common price ranges
- The peak represents the most frequent house prices
- Long tail indicates some very high-priced houses
- Helps understand typical price ranges in the market
""")

st.write("### How Quality Affects Price")
fig2, ax2 = plt.subplots()
sns.boxplot(x=data['OverallQual'], y=data['SalePrice'], palette='Blues', ax=ax2)
ax2.set_xlabel('Quality Rating (1-10)')
ax2.set_ylabel('Price ($)')
st.pyplot(fig2)

st.markdown("""
**What this tells us:**
- Clear relationship: Higher quality = Higher price
- Each box shows price range for that quality level
- Middle line in each box is the median price
- Box shows where 50% of prices fall (25th-75th percentile)
- Whiskers show typical range (excluding outliers)
- Outliers (dots) are unusually priced homes
""")

st.markdown("---")

st.subheader("How to Interpret These Charts")
st.markdown("""
1. **Compare your predicted price** to the distributions shown
2. **Higher quality homes** (7-10 rating) command premium prices
3. Most homes fall in the **main cluster** of the price distribution
4. If your prediction seems very high/low compared to these charts, double-check your inputs
""")
