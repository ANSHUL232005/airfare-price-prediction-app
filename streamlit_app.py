import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import os

# Set page configuration
st.set_page_config(
    page_title="Airfare Price Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton > button {
        background-color: #FF6B6B;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #FF5252;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF6B6B;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load trained model and feature names
@st.cache_resource
def load_model():
    MODEL_PATH = "airfare_model.pkl"
    FEATURES_PATH = "feature_names.pkl"
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        st.error("Model files not found. Please run the training script first.")
        st.stop()
    
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, feature_names

# Load raw data for extracting unique values
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('raw_data.csv', low_memory=False)
        data = data[data['price'] != 'Unknown']
        data = data[data['from'] != 'Unknown']
        data = data[data['to'] != 'Unknown']
        data = data[data['airline'] != 'Unknown']
        
        # Remove rows with NaN values in critical columns
        data = data.dropna(subset=['price', 'from', 'to', 'airline', 'stop', 'Class'])
        
        # Convert columns to string to handle any remaining issues
        data['airline'] = data['airline'].astype(str)
        data['from'] = data['from'].astype(str)
        data['to'] = data['to'].astype(str)
        data['stop'] = data['stop'].astype(str)
        data['Class'] = data['Class'].astype(str)
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Main title
st.title("✈️ Airfare Price Prediction System")
st.markdown("---")

# Load model and data
model, feature_names = load_model()
data = load_data()

if data is not None:
    # Sidebar for user inputs
    st.sidebar.header("🛫 Flight Details")
    
    # Get unique values from the dataset
    unique_airlines = sorted(data['airline'].unique())
    unique_from = sorted(data['from'].unique())
    unique_to = sorted(data['to'].unique())
    unique_stops = sorted(data['stop'].unique())
    unique_classes = sorted(data['Class'].unique())
    
    # User inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Route Information")
        departure_city = st.selectbox("Departure City", unique_from, index=0)
        arrival_city = st.selectbox("Arrival City", unique_to, index=1)
        airline = st.selectbox("Airline", unique_airlines, index=0)
    
    with col2:
        st.subheader("🕐 Flight Details")
        flight_date = st.date_input("Flight Date", value=datetime.now() + timedelta(days=30))
        departure_time = st.time_input("Departure Time", value=datetime.now().time())
        arrival_time = st.time_input("Arrival Time", value=datetime.now().time())
        flight_class = st.selectbox("Flight Class", unique_classes, index=0)
        stops = st.selectbox("Stops", unique_stops, index=0)
    
    # Calculate features
    day_of_week = flight_date.weekday()
    month = flight_date.month
    day = flight_date.day
    
    # Convert times to minutes
    dep_time_minutes = departure_time.hour * 60 + departure_time.minute
    arr_time_minutes = arrival_time.hour * 60 + arrival_time.minute
    
    # Calculate flight duration (handle overnight flights)
    if arr_time_minutes >= dep_time_minutes:
        flight_duration_minutes = arr_time_minutes - dep_time_minutes
    else:
        flight_duration_minutes = (24 * 60) - dep_time_minutes + arr_time_minutes
    
    # Encode categorical variables using the same approach as training
    # We need to create label encoders for each categorical column
    le_airline = LabelEncoder()
    le_from = LabelEncoder()
    le_to = LabelEncoder()
    le_stop = LabelEncoder()
    le_class = LabelEncoder()
    
    # Fit the encoders on the full dataset
    le_airline.fit(data['airline'].astype(str))
    le_from.fit(data['from'].astype(str))
    le_to.fit(data['to'].astype(str))
    le_stop.fit(data['stop'].astype(str))
    le_class.fit(data['Class'].astype(str))
    
    # Encode the user inputs
    try:
        airline_encoded = le_airline.transform([airline])[0]
        from_encoded = le_from.transform([departure_city])[0]
        to_encoded = le_to.transform([arrival_city])[0]
        stop_encoded = le_stop.transform([stops])[0]
        class_encoded = le_class.transform([flight_class])[0]
    except ValueError as e:
        st.error(f"Error encoding categorical variables: {str(e)}")
        st.stop()
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'airline': [airline_encoded],
        'from': [from_encoded],
        'stop': [stop_encoded],
        'to': [to_encoded],
        'Class': [class_encoded],
        'day_of_week': [day_of_week],
        'month': [month],
        'day': [day],
        'dep_time_minutes': [dep_time_minutes],
        'arr_time_minutes': [arr_time_minutes],
        'flight_duration_minutes': [flight_duration_minutes]
    })
    
    # Ensure columns are in the same order as training
    input_data = input_data[feature_names]
    
    # Display input summary
    st.subheader("📋 Flight Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Route", f"{departure_city} → {arrival_city}")
        st.metric("Airline", airline)
    
    with col2:
        st.metric("Date", flight_date.strftime("%B %d, %Y"))
        st.metric("Departure", departure_time.strftime("%H:%M"))
    
    with col3:
        st.metric("Duration", f"{flight_duration_minutes // 60}h {flight_duration_minutes % 60}m")
        st.metric("Class", flight_class)
    
    # Prediction
    if st.button("🔮 Predict Price", key="predict_button"):
        try:
            prediction = model.predict(input_data)
            predicted_price = prediction[0]
            
            # Display prediction
            st.markdown("---")
            st.subheader("💰 Price Prediction")
            
            # Create a nice prediction box
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="color: #FF6B6B; margin: 0;">₹{predicted_price:,.2f}</h2>
                <p style="margin: 0; color: #666;">Predicted Airfare Price</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"💡 **Tip**: Prices may vary based on booking time and availability")
            
            with col2:
                if predicted_price > 10000:
                    st.warning("⚠️ This seems like a premium flight")
                elif predicted_price < 5000:
                    st.success("✅ This looks like a good deal!")
                else:
                    st.info("ℹ️ This is a moderately priced flight")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Data visualization section
    st.markdown("---")
    st.subheader("📊 Data Insights")
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Distribution by Airlines")
        # Process price data
        viz_data = data.copy()
        viz_data['price'] = viz_data['price'].str.replace(',', '').astype(float)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        top_airlines = viz_data['airline'].value_counts().head(8).index
        airline_data = viz_data[viz_data['airline'].isin(top_airlines)]
        
        sns.boxplot(data=airline_data, x='airline', y='price', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title('Price Distribution by Top Airlines')
        ax.set_ylabel('Price (₹)')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Flight Count by Routes")
        route_counts = viz_data.groupby(['from', 'to']).size().reset_index(name='count')
        route_counts = route_counts.sort_values('count', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        route_counts['route'] = route_counts['from'] + ' → ' + route_counts['to']
        sns.barplot(data=route_counts, x='count', y='route', ax=ax)
        ax.set_title('Top 10 Routes by Flight Count')
        ax.set_xlabel('Number of Flights')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit | Airfare Price Prediction System")

else:
    st.error("Could not load the dataset. Please check if 'raw_data.csv' exists.")
