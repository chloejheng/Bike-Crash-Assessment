import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv("bike_crash_updated.csv")
df_encoded = pd.read_csv("bike_crash_encoded.csv")

# Find emoji here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Bike Risk Assessment", page_icon=":🚲:")

# Create the Streamlit app
def main():
    st.title("Austin Bike Crash Risk Analysis")
    st.write("This app analyzes bike crash data in Austin and predicts the risk of a crash occurring based on user input.")

    # Display dataset overview
    st.subheader("Dataset Overview")
    st.write(df.head())
    st.write(f"Total crashes: {len(df)}")

    # Visualize crash distribution by hour
    st.subheader("Crash Distribution by Hour")
    hour_dist = df['Hour'].value_counts().sort_index()
    st.bar_chart(hour_dist)

    # Visualize crash severity by hour
    st.subheader("Crash Severity by Hour")
    severity_by_hour = df.groupby(['Hour', 'Crash Severity']).size().unstack()
    severity_options = ['Incapacitating Injury', 'Killed', 'Non-Incapacitating Injury', 'Not Injured', 'Possible Injury']
    selected_severity = st.multiselect("Select Crash Severity", options=severity_options, default=severity_options)
    st.area_chart(severity_by_hour[selected_severity])
    
    # User input for all variables for risk prediction
    st.subheader("Enter Crash Details")
    active_school_zone = st.selectbox("Active School Zone", options=df['Active School Zone Flag'].unique())
    crash_time = st.text_input("Crash Time (HH:MM)", value="12:00")
    am_pm = st.radio("AM/PM", options=["AM", "PM"])
    day_of_week = st.selectbox("Day of Week", options=df['Day of Week'].unique())
    roadway_part = st.selectbox("Roadway Part", options=df['Roadway Part'].unique())
    speed_limit = st.slider("Speed Limit", min_value=df['Speed Limit'].min(), max_value=df['Speed Limit'].max(), value=30)
    surface_condition = st.selectbox("Surface Condition", options=df['Surface Condition'].unique())
    person_helmet = st.selectbox("Person Helmet Binary", options=df['Person Helmet Binary'].unique())
    
    # Convert crash time input to 24-hour format
    crash_time_parts = crash_time.split(":")
    hour = int(crash_time_parts[0])
    if am_pm == "PM" and hour != 12:
        hour += 12
    if am_pm == "AM" and hour == 12:
        hour = 0

 # Add a button to generate the prediction result
    if st.button("Generate Prediction"):
        # Prepare data for prediction
        X = df_encoded[['Active School Zone Flag', 'Day of Week', 'Roadway Part', 'Speed Limit', 'Surface Condition', 'Person Helmet Binary', 'Hour']]
        y = df_encoded['Crash Severity Binary']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train decision tree model
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Make prediction based on user input
        user_input = pd.DataFrame({
            'Active School Zone Flag': [active_school_zone],
            'Day of Week': [day_of_week],
            'Roadway Part': [roadway_part],
            'Speed Limit': [speed_limit],
            'Surface Condition': [surface_condition],
            'Person Helmet Binary': [person_helmet],
            'Hour': [hour]
        })

        # Encode the user input variables
        user_input['Active School Zone Flag'] = user_input['Active School Zone Flag'].map({'Yes': 1, 'No': 0})
        user_input['Day of Week'] = user_input['Day of Week'].map({'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6})
        user_input['Roadway Part'] = user_input['Roadway Part'].map({'Main/Proper Lane': 0, 'Service/Frontage Road': 1, 'Other (Explain In Narrative)': 2, 'Entrance/On Ramp': 3})
        user_input['Surface Condition'] = user_input['Surface Condition'].map({'Dry': 0, 'Wet': 1, 'Sand, Mud, Dirt': 2, 'Other (Explain In Narrative)': 3, 'Unknown': 4, 'Ice': 5, 'Standing Water': 6})
        user_input['Person Helmet Binary'] = user_input['Person Helmet Binary'].map({'Not Wear': 0, 'Wear': 1})

        prediction = model.predict(user_input)[0]
        
        # Display prediction result
        st.subheader("Crash Risk Prediction")
        if prediction < 0.5:
            st.success(f"The predicted crash risk for the given details is: Low ({prediction:.2f})")
        else:
            st.warning(f"The predicted crash risk for the given details is: High ({prediction:.2f})")
           
        # Get the number of crash cases at the specified hour
        crash_count = df[df['Hour'] == hour].shape[0]
        if am_pm == "PM":
            st.write(f"<span style='color:red'>Number of crash cases happened at {crash_time} PM: {crash_count} cases </span>", unsafe_allow_html=True)
        else:
            st.write(f"<span style='color:red'>Number of crash cases happened at {crash_time} AM: {crash_count} cases </span>", unsafe_allow_html=True)
        
        # Display top three feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        feature_names = X.columns
        st.subheader("Top Three Feature Importances")
        for i in range(3):
            st.write(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.2f}")
            
        # Plot feature importances
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        feature_names = X.columns
        indices = np.argsort(importances)[::-1]
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), feature_names[indices], rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importances')
        st.pyplot(plt)
            
        # Display model performance metrics
        st.subheader("Model Performance Metrics")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy:.2f}")
        st.write("Confusion Matrix:")
        st.write(conf_matrix)

        # Provide safety recommendations
        if prediction >= 0.5:
            st.warning("High crash risk predicted. Please exercise caution and consider the following safety measures:")
            st.write("- Wear a helmet")
            st.write("- Use designated bike lanes or paths")
            st.write("- Be visible and use lights during low light conditions")
        else:
            st.success("Lower crash risk predicted. Enjoy your ride but still prioritize safety:")
            st.write("- Always wear a helmet")
            st.write("- Follow traffic rules and signals")
            st.write("- Stay alert and watch for road hazards")
            
    # Visualize crash severity vs helmet usage
    st.subheader("Crash Severity vs Helmet Usage")
    helmet_severity = pd.crosstab(df['Person Helmet'], df['Crash Severity'])
    st.bar_chart(helmet_severity)
    
if __name__ == '__main__':
    main()
