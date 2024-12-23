import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Telecom Analysis Dashboard", layout="wide")

# Load dataset
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# Path to dataset
file_path = "C:\\Users\\Hasan\\Desktop\\data science folder\\Copy of Week2_challenge_data_source(CSV).csv"
data = load_data(file_path)

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.selectbox(
    "Select Analysis Section",
    [
        "User Overview Analysis",
        "User Engagement Analysis",
        "User Experience Analysis",
        "User Satisfaction Analysis"
    ]
)

# Initialize global variables
if "engagement_data" not in st.session_state:
    st.session_state.engagement_data = None
    st.session_state.engagement_kmeans = None

if "experience_data" not in st.session_state:
    st.session_state.experience_data = None
    st.session_state.experience_kmeans = None

# User Overview Analysis
if section == "User Overview Analysis":
    st.title("User Overview Analysis")

    # Top 10 handsets
    st.subheader("Top 10 Handsets")
    top_handsets = data['Handset Type'].value_counts().head(10)
    st.bar_chart(top_handsets)

    # Top 3 manufacturers
    st.subheader("Top 3 Handset Manufacturers")
    top_manufacturers = data['Handset Manufacturer'].value_counts().head(3)
    st.bar_chart(top_manufacturers)

    # Top 5 handsets per top 3 manufacturers
    st.subheader("Top 5 Handsets per Top 3 Manufacturers")
    for manufacturer in top_manufacturers.index:
        st.write(f"**Manufacturer: {manufacturer}**")
        top_handsets = data[data['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
        st.write(top_handsets)

    # Aggregate per user
    st.subheader("Aggregate User Data")
    user_data = data.groupby('MSISDN/Number').agg({
        'Bearer Id': 'count',
        'Dur. (ms)': 'sum',
        'Total DL (Bytes)': 'sum',
        'Total UL (Bytes)': 'sum',
    }).rename(columns={
        'Bearer Id': 'xDR Sessions',
        'Dur. (ms)': 'Total Session Duration',
        'Total DL (Bytes)': 'Total Download',
        'Total UL (Bytes)': 'Total Upload'
    })
    st.write(user_data.head(10))

    # Correlation analysis
    st.subheader("Correlation Analysis")
    correlation_columns = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)',
                           'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    correlation_matrix = data[correlation_columns].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    st.pyplot(plt)

# User Engagement Analysis
elif section == "User Engagement Analysis":
    st.title("User Engagement Analysis")

    # Engagement metrics
    st.subheader("Engagement Metrics")
    st.session_state.engagement_data = data.groupby('MSISDN/Number').agg({
        'Bearer Id': 'count',
        'Dur. (ms)': 'sum',
        'Total DL (Bytes)': 'sum',
        'Total UL (Bytes)': 'sum',
    }).rename(columns={
        'Bearer Id': 'Session Frequency',
        'Dur. (ms)': 'Session Duration',
        'Total DL (Bytes)': 'Total Download',
        'Total UL (Bytes)': 'Total Upload'
    }).dropna()
    st.session_state.engagement_data.reset_index(inplace=True)
    st.write(st.session_state.engagement_data.head(10))

    # Top 3 most-used applications
    st.subheader("Top 3 Most-Used Applications")
    application_data = data[['Social Media DL (Bytes)', 'Youtube DL (Bytes)', 'Google DL (Bytes)']].sum()
    st.bar_chart(application_data)

    # K-means clustering
    st.subheader("Engagement Clustering")
    st.session_state.engagement_kmeans = KMeans(n_clusters=3, random_state=42)
    engagement_clusters = st.session_state.engagement_kmeans.fit_predict(
        st.session_state.engagement_data[['Session Frequency', 'Session Duration', 'Total Download', 'Total Upload']]
    )
    st.session_state.engagement_data['Cluster'] = engagement_clusters
    st.write(st.session_state.engagement_data.groupby('Cluster').mean())
    # User Experience Analysis
elif section == "User Experience Analysis":
    st.title("User Experience Analysis")

    # Key metrics
    st.subheader("Experience Metrics")
    st.session_state.experience_data = data[['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 
                                             'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']]
    st.session_state.experience_data = st.session_state.experience_data.dropna()
    st.write(st.session_state.experience_data.describe())

    # Throughput distribution per handset type
    st.subheader("Throughput Distribution by Handset Type")
    throughput_data = data.groupby('Handset Type')[['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']].mean()
    st.bar_chart(throughput_data)

    # K-means clustering
    st.subheader("Experience Clustering")
    if st.session_state.experience_data is not None:
        st.session_state.experience_kmeans = KMeans(n_clusters=3, random_state=42)
        experience_clusters = st.session_state.experience_kmeans.fit_predict(st.session_state.experience_data)
        st.session_state.experience_data['Cluster'] = experience_clusters
        st.write(st.session_state.experience_data.groupby('Cluster').mean())
    else:
        st.error("Experience data is not available!")


# User Satisfaction Analysis
elif section == "User Satisfaction Analysis":
    st.title("User Satisfaction Analysis")

    # Compute engagement and experience scores
    st.subheader("Satisfaction Metrics")
    if st.session_state.engagement_data is not None and st.session_state.engagement_kmeans is not None:
        engagement_cluster_centers = st.session_state.engagement_kmeans.cluster_centers_
        st.session_state.engagement_data['Engagement Score'] = st.session_state.engagement_data.apply(
            lambda row: euclidean(
                row[['Session Frequency', 'Session Duration', 'Total Download', 'Total Upload']],
                engagement_cluster_centers[0]
            ), axis=1
        )

        st.write(st.session_state.engagement_data.head(10))
    else:
        st.error("Engagement clustering must be completed first!")
