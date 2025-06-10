import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load and preprocess data
@st.cache_data
def load_data_and_train():
    df = pd.read_csv(r"C:\Users\HP\Downloads\bank+marketing\bank\bank-full.csv", sep=';')

    label_encoders = {}
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    X = df_encoded.drop('y', axis=1)
    y = df_encoded['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, label_encoders, df

# Load trained model and encoders
model, label_encoders, original_df = load_data_and_train()

# Sidebar Navigation
st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Choose Page", ["Home", "Raw Data", "Summary", "Graphs & Charts"])

st.title("🏦 Bank Term Deposit Predictor")

# Home Page - Prediction Interface
if page == "Home":
    user_input = {}

    for col in original_df.columns[:-1]:  # Exclude target 'y'
        if col == "pdays":
            user_input[col] = st.slider(
                f"{col} (999 = never contacted before)",
                min_value=original_df[col].min(),
                max_value=original_df[col].max(),
                value=999
            )
        elif col == "poutcome":
            options = ['success', 'failure', 'other', 'unknown']
            selected = st.selectbox("poutcome", options)
            user_input[col] = label_encoders[col].transform([selected])[0]
        elif original_df[col].dtype == 'object':
            options = sorted(original_df[col].unique())
            selected = st.selectbox(f"{col}", options)
            user_input[col] = label_encoders[col].transform([selected])[0]
        else:
            min_val = int(original_df[col].min())
            max_val = int(original_df[col].max())
            default_val = int(original_df[col].mean())
            user_input[col] = st.slider(
                f"{col}", min_value=min_val, max_value=max_val, value=default_val
            )

    if st.button("🔍 Predict"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        result = label_encoders['y'].inverse_transform([prediction])[0]

        if result == 'yes':
            st.success("✅ The client is likely to **subscribe** to a term deposit.")
        else:
            st.error("❌ The client is **not likely to subscribe** to a term deposit.")

# Raw Data Page
elif page == "Raw Data":
    st.subheader("📊 Raw Dataset")
    st.dataframe(original_df)

# Summary Page
elif page == "Summary":
    st.subheader("📈 Summary Statistics")
    st.write(original_df.describe())

# Graphs & Charts Page
elif page == "Graphs & Charts":
    st.subheader("📊 Visual Insights")
    st.bar_chart(original_df['age'].value_counts().sort_index())
    st.line_chart(original_df['balance'].sort_values().reset_index(drop=True))
