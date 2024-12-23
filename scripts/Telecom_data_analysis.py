import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns
# Provide the file path
file_path = "C:\\Users\\Hasan\\Desktop\\data science folder\\Copy of Week2_challenge_data_source(CSV).csv"  

# Load the CSV file into a DataFrame
data = pd.read_csv(file_path)
# Ensure all columns are displayed
pd.set_option('display.max_columns', None)

# Optional: Display all column names
print("Column Names:", data.columns.tolist())

# Display the first few rows
print(data.head()
print(data.info()) # To Identify Data types and non-null counts
print(data.describe()) # To Understand Statistical summary of numerical columns
# Count missing values in each column
print(data.isnull().sum())
msno.matrix(data)  # Visualize missing data as a matrix
plt.show()

msno.heatmap(data)  # Correlation of missing values
plt.show()
#Compute Z-scores to identify outliers
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
z_scores = data[numerical_cols].apply(zscore)

# Detect outliers with Z-score threshold
outliers_z = (z_scores.abs() > 3).sum()
print(outliers_z)
numeric_columns = data.select_dtypes(include=['number']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

data['Start'] = data['Start'].fillna('Unknown')
data['End'] = data['End'].fillna('Unknown')
data['Last Location Name'] = data['Last Location Name'].fillna('Unknown')
# Verify
print("\nMissing values after filling:")
print(data.isnull().sum())
#Get the top 10 most used handsets
top_10_handsets = handset_counts.head(10)
print("Top 10 Handsets Used by Customers:")
print(top_10_handsets)
#Draw a bar chart to visualize the top 10 handsets:
# Plot the top 10 handsets
top_10_handsets.plot(kind='bar', color='skyblue')
plt.title("Top 10 Handsets Used by Customers")
plt.xlabel("Handset Model")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45, ha="right")
plt.show()
# Extract manufacturer names
manufacturers = handset_counts.index.str.split().str[0]  # Extract first word (e.g., "Apple", "Huawei", "Samsung")

# Create a DataFrame with manufacturer counts
manufacturer_counts = handset_counts.groupby(manufacturers).sum().sort_values(ascending=False)

# Get the top 3 manufacturers
top_3_manufacturers = manufacturer_counts.head(3)

# Display results
print("Top 3 Handset Manufacturers:")
print(top_3_manufacturers)
# Plot the results
top_3_manufacturers.plot(kind='bar', color='skyblue', figsize=(8, 5))
plt.title("Top 3 Handset Manufacturers")
plt.xlabel("Manufacturer")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
# Extract manufacturer names
manufacturers = handset_counts.index.str.split().str[0]  # Extract first word as manufacturer

# Create a DataFrame with manufacturers and handsets
handset_data = pd.DataFrame({
    'Manufacturer': manufacturers,
    'Handset': handset_counts.index,
    'Count': handset_counts.values
})

# Identify top 3 manufacturers
top_3_manufacturers = (
    handset_data.groupby('Manufacturer')['Count']
    .sum()
    .sort_values(ascending=False)
    .head(3)
    .index
)

# Filter handsets belonging to the top 3 manufacturers
filtered_handset_data = handset_data[handset_data['Manufacturer'].isin(top_3_manufacturers)]
# Identify top 5 handsets for each manufacturer
top_5_per_manufacturer = (
    filtered_handset_data.groupby('Manufacturer', group_keys=False)
    .apply(lambda x: x.sort_values(by='Count', ascending=False).head(5))
)

# Display results
print("Top 5 Handsets per Top 3 Manufacturers:")
print(top_5_per_manufacturer)
# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_5_per_manufacturer,
    x='Count',
    y='Handset',
    hue='Manufacturer',
    dodge=False
)
plt.title("Top 5 Handsets per Top 3 Manufacturers")
plt.xlabel("Count")
plt.ylabel("Handset")
plt.tight_layout()
plt.show()
# 1. Aggregate the number of xDR sessions per user
num_sessions = data.groupby('IMSI')['Bearer Id'].nunique().rename('Number_of_Sessions')

# 2. Aggregate the total session duration per user
total_duration = data.groupby('IMSI')['Dur. (ms)'].sum().rename('Total_Session_Duration_ms')

# 3. Aggregate total DL and UL data per user
total_download = data.groupby('IMSI')['Total DL (Bytes)'].sum().rename('Total_Download_Bytes')
total_upload = data.groupby('IMSI')['Total UL (Bytes)'].sum().rename('Total_Upload_Bytes')

# 4. Aggregate total data volume for each application per user
application_columns = [
    'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
    'Google DL (Bytes)', 'Google UL (Bytes)',
    'Email DL (Bytes)', 'Email UL (Bytes)',
    'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
    'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
    'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
    'Other DL (Bytes)', 'Other UL (Bytes)'
]

# Calculate total data volume for each application
for app in ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']:
    data[f'{app}_Total_Bytes'] = data[f'{app} DL (Bytes)'] + data[f'{app} UL (Bytes)']

app_columns = [f'{app}_Total_Bytes' for app in ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']]
total_app_data = data.groupby('IMSI')[app_columns].sum()
# Combine all results into one DataFrame
user_summary = pd.concat([num_sessions, total_duration, total_download, total_upload, total_app_data], axis=1)

# Display the final user summary
print("Per-User Aggregated Summary:")
print(user_summary)
porting
decile_summary.to_csv("decile_summary.csv", index=False)
# Plot total data volume per decile
plt.figure(figsize=(10, 6))
plt.bar(decile_summary['Decile'], decile_summary['Total_Data_Volume'], color='skyblue')
plt.title("Total Data Volume per Decile Class")
plt.xlabel("Decile Class")
plt.ylabel("Total Data Volume (Bytes)")
plt.xticks(decile_summary['Decile'])
plt.tight_layout()
plt.show()
#Non-Graphical Univariate Analysis
# Select quantitative variables
quantitative_cols = data.select_dtypes(include=['float64', 'int64']).columns

# Calculate dispersion parameters
dispersion_metrics = pd.DataFrame({
    'Range': data[quantitative_cols].max() - data[quantitative_cols].min(),
    'Variance': data[quantitative_cols].var(),
    'Standard Deviation': data[quantitative_cols].std(),
    'IQR': data[quantitative_cols].quantile(0.75) - data[quantitative_cols].quantile(0.25)
})

# Display results
print("Dispersion Metrics for Quantitative Variables:")
print(dispersion_metrics)
#Graphical Univariate Analysis
# Histogram for Total DL (Bytes)
plt.figure(figsize=(8, 5))
sns.histplot(data['Total DL (Bytes)'], bins=30, kde=True, color='skyblue')
plt.title("Distribution of Total Download Data (Bytes)")
plt.xlabel("Total Download Data (Bytes)")
plt.ylabel("Frequency")
plt.show()

# Box plot for Session Duration
plt.figure(figsize=(8, 5))
sns.boxplot(x=data['Dur. (ms)'], color='lightgreen')
plt.title("Box Plot of Session Duration")
plt.xlabel("Duration (ms)")
plt.show()

# Bar plot for Handset Manufacturer
top_manufacturers = data['Handset Manufacturer'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_manufacturers.index, y=top_manufacturers.values, palette="viridis")
plt.title("Top 10 Handset Manufacturers")
plt.xlabel("Manufacturer")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()
#Boxplot for Total Data Usage by Application
#List of applications for analysis
applications = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Youtube DL (Bytes)', 
                'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']

# Reshape data for plotting
data_melted = data[applications].melt(var_name='Application', value_name='Data Usage')

# Create a boxplot to compare data usage across applications
plt.figure(figsize=(12, 6))
sns.boxplot(x='Application', y='Data Usage', data=data_melted)
plt.title('Total Data Usage by Application')
plt.xticks(rotation=45)
plt.show()
# Scatter Plot for Correlation Between Application Data Usage and Total Data Usage
# Pearson correlation (for linear relationships)
pearson_corr = data[['Social Media DL (Bytes)', 'Total DL (Bytes)']].corr(method='pearson')
print("Pearson Correlation between Social Media and Total Data Usage:")
print(pearson_corr)

# Spearman correlation (for non-linear relationships)
spearman_corr = data[['Social Media DL (Bytes)', 'Total DL (Bytes)']].corr(method='spearman')
print("Spearman Correlation between Social Media and Total Data Usage:")
print(spearman_corr)
#Strip leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Calculate the total data used (Total DL + Total UL)
data['Total_Data'] = data['Total DL (Bytes)'] + data['Total UL (Bytes)']

# Verify the new column was created
print(data[['Total DL (Bytes)', 'Total UL (Bytes)', 'Total_Data']].head())
from scipy import stats

#Statistical Interpretation
# Strip leading/trailing spaces in the column names
data.columns = data.columns.str.strip()

# Calculate Total Data Usage (Total DL + Total UL)
data['Total_Data'] = data['Total DL (Bytes)'] + data['Total UL (Bytes)']

# List of application columns to explore
applications = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Youtube DL (Bytes)', 
                'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']

# Reshape the data for plotting
data_melted = data[applications].melt(var_name='Application', value_name='Data Usage')

# Boxplot to compare the distribution of data usage across applications
plt.figure(figsize=(12, 6))
sns.boxplot(x='Application', y='Data Usage', data=data_melted)
plt.title('Total Data Usage by Application')
plt.xticks(rotation=45)
plt.show()

# Scatter plot for each application data usage vs total data usage
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Social Media DL (Bytes)', y='Total_Data', data=data)
plt.title('Social Media Data Usage vs Total Data Usage')
plt.show()
# Compute Pearson correlation (for linear relationships)
pearson_corr = data[['Social Media DL (Bytes)', 'Total_Data']].corr(method='pearson')
print("Pearson Correlation between Social Media and Total Data Usage:")
print(pearson_corr)

# Compute Spearman correlation (for non-linear relationships)
spearman_corr = data[['Social Media DL (Bytes)', 'Total_Data']].corr(method='spearman')
print("Spearman Rank Correlation between Social Media and Total Data Usage:")
print(spearman_corr)

# Perform ANOVA for Social Media vs Google vs YouTube vs Netflix
f_stat, p_value = stats.f_oneway(data['Social Media DL (Bytes)'], 
                                 data['Google DL (Bytes)'], 
                                 data['Youtube DL (Bytes)'], 
                                 data['Netflix DL (Bytes)'], 
                                 data['Gaming DL (Bytes)'], 
                                 data['Other DL (Bytes)'])
print(f"ANOVA F-statistic: {f_stat}, p-value: {p_value}")
# Drop non-numeric and identifier columns
columns_to_drop = ['Bearer Id', 'Start', 'End', 'Start ms', 'End ms', 'Last Location Name', 'Handset Manufacturer', 'Handset Type']
data_numeric = data.drop(columns=columns_to_drop, errors='ignore')

# Check for and handle missing values
data_numeric = data_numeric.fillna(data_numeric.mean())

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_numeric)

# Perform PCA
pca = PCA(n_components=0.95)  # Retain components that explain 95% of the variance
pca_data = pca.fit_transform(scaled_data)

# Results summary
explained_variance = pca.explained_variance_ratio_
cumulative_variance = pca.explained_variance_ratio_.cumsum()

# Save or display results
print(f"Number of components retained: {pca.n_components_}")
print(f"Explained variance by each component: {explained_variance}")
print(f"Cumulative variance explained: {cumulative_variance}")

# Save transformed data to a new CSV file
pca_data = pd.DataFrame(pca_data, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
pca_data.to_csv("dataset_pca.csv", index=False)
print("PCA-transformed data saved to 'dataset_pca.csv'")
# Ensure required columns exist
required_columns = ['IMSI', 'Start', 'End', 'Dur. (ms)', 'Total UL (Bytes)', 'Total DL (Bytes)']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Missing required column: {col}")

# Calculate Session Duration (ms)
# Option 1: Use 'Dur. (ms)' directly
data['Session Duration (ms)'] = data['Dur. (ms)']

# Option 2: Calculate from 'Start' and 'End'
# Ensure 'Start' and 'End' are datetime objects
data['Start'] = pd.to_datetime(data['Start'], errors='coerce')
data['End'] = pd.to_datetime(data['End'], errors='coerce')
data['Session Duration (ms)'] = (data['End'] - data['Start']).dt.total_seconds() * 1000

# Calculate Total Traffic (Upload + Download)
data['Total Traffic (Bytes)'] = data['Total UL (Bytes)'] + data['Total DL (Bytes)']

# Aggregating Metrics Per User
engagement_metrics = data.groupby('IMSI').agg({
    'Start': 'count',  # Session frequency
    'Session Duration (ms)': 'mean',  # Average session duration
    'Total Traffic (Bytes)': 'sum'  # Total traffic per user
}).reset_index()

# Rename columns for clarity
engagement_metrics.rename(columns={'Start': 'Sessions Frequency'}, inplace=True)
print(engagement_metrics)
print(engagement_metrics.to_string())
# Aggregate metrics per customer (MSISDN)
customer_metrics = data.groupby('MSISDN/Number').agg({
    'Bearer Id': 'count',  # Session frequency
    'Dur. (ms)': 'sum',    # Total session duration
    'Total DL (Bytes)': 'sum',  # Total download traffic
    'Total UL (Bytes)': 'sum'   # Total upload traffic
}).rename(columns={
    'Bearer Id': 'Session_Frequency',
    'Dur. (ms)': 'Total_Session_Duration',
    'Total DL (Bytes)': 'Total_Download_Bytes',
    'Total UL (Bytes)': 'Total_Upload_Bytes'
})

# Calculate total traffic (download + upload)
customer_metrics['Total_Traffic'] = customer_metrics['Total_Download_Bytes'] + customer_metrics['Total_Upload_Bytes']

# Find top 10 customers for each metric
top_10_sessions = customer_metrics.sort_values(by='Session_Frequency', ascending=False).head(10)
top_10_duration = customer_metrics.sort_values(by='Total_Session_Duration', ascending=False).head(10)
top_10_traffic = customer_metrics.sort_values(by='Total_Traffic', ascending=False).head(10)

# Display results
print("Top 10 Customers by Session Frequency:")
print(top_10_sessions)

print("\nTop 10 Customers by Total Session Duration:")
print(top_10_duration)
print("\nTop 10 Customers by Total Traffic (Download + Upload):")
print(top_10_traffic)
# Aggregate metrics per customer (MSISDN/Number)
customer_metrics = data.groupby('MSISDN/Number').agg({
    'Bearer Id': 'count',  # Session frequency
    'Dur. (ms)': 'sum',    # Total session duration
    'Total DL (Bytes)': 'sum',  # Total download traffic
    'Total UL (Bytes)': 'sum'   # Total upload traffic
}).rename(columns={
    'Bearer Id': 'Session_Frequency',
    'Dur. (ms)': 'Total_Session_Duration',
    'Total DL (Bytes)': 'Total_Download_Bytes',
    'Total UL (Bytes)': 'Total_Upload_Bytes'
})

# Calculate total traffic (download + upload)
customer_metrics['Total_Traffic'] = customer_metrics['Total_Download_Bytes'] + customer_metrics['Total_Upload_Bytes']

# 1. Normalize metrics
scaler = MinMaxScaler()
normalized_metrics = scaler.fit_transform(customer_metrics[['Session_Frequency', 'Total_Session_Duration', 'Total_Traffic']])
normalized_df = pd.DataFrame(normalized_metrics, columns=['Normalized_Session_Frequency', 'Normalized_Total_Session_Duration', 'Normalized_Total_Traffic'])

# 2. Run k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(normalized_metrics)

# Add cluster labels to the original DataFrame
customer_metrics['Engagement_Cluster'] = clusters
# 3. Analyze clusters
cluster_summary = customer_metrics.groupby('Engagement_Cluster').mean()

# Display results
print("Cluster Summary:")
print(cluster_summary)
# 4. Visualize clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=normalized_df['Normalized_Total_Traffic'],
    y=normalized_df['Normalized_Total_Session_Duration'],
    hue=clusters,
    palette='viridis'
)
plt.title("Customer Engagement Clusters")
plt.xlabel("Normalized Total Traffic")
plt.ylabel("Normalized Total Session Duration")
plt.show()
# Compute aggregation metrics per cluster
cluster_aggregates = customer_metrics.groupby('Engagement_Cluster').agg({
    'Session_Frequency': ['min', 'max', 'mean', 'sum'],
    'Total_Session_Duration': ['min', 'max', 'mean', 'sum'],
    'Total_Traffic': ['min', 'max', 'mean', 'sum']
})

# Flatten multi-level column names
cluster_aggregates.columns = ['_'.join(col) for col in cluster_aggregates.columns]
cluster_aggregates.reset_index(inplace=True)

# Display cluster aggregates
print("Cluster Aggregates:")
print(cluster_aggregates)
# Visualization of metrics
plt.figure(figsize=(12, 6))
metrics = ['Session_Frequency', 'Total_Session_Duration', 'Total_Traffic']

for i, metric in enumerate(metrics):
    plt.subplot(1, 3, i + 1)
    sns.barplot(
        data=cluster_aggregates,
        x='Engagement_Cluster',
        y=f'{metric}_sum',
        palette='viridis'
    )
    plt.title(f'Total {metric} by Cluster')
    plt.xlabel('Engagement Cluster')
    plt.ylabel(f'Total {metric}')

plt.tight_layout()
plt.show()

# Visualizing Min, Max, and Average
plt.figure(figsize=(12, 6))
for i, metric in enumerate(metrics):
    plt.subplot(1, 3, i + 1)
    cluster_aggregates.plot(
        x='Engagement_Cluster',
        y=[f'{metric}_min', f'{metric}_max', f'{metric}_mean'],
        kind='bar',
        figsize=(12, 6),
        title=f'{metric}: Min, Max, and Mean by Cluster',
        ax=plt.subplot(1, 3, i + 1),
    )
    plt.xlabel('Engagement Cluster')
    plt.ylabel(metric)
    plt.legend(['Min', 'Max', 'Mean'])

plt.tight_layout()
plt.show()
# Applications to analyze
applications = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']

# Create a DataFrame to store top 10 users per application
top_users_per_application = {}

# Loop through each application to calculate total traffic and find top 10 users
for app in applications:
    # Calculate total traffic (DL + UL) for the application
    data[f'{app}_Total_Traffic'] = data[f'{app} DL (Bytes)'] + data[f'{app} UL (Bytes)']
    
    # Aggregate total traffic per user (MSISDN/Number)
    app_traffic = data.groupby('MSISDN/Number')[f'{app}_Total_Traffic'].sum()
    
    # Sort and select top 10 users
    top_10_users = app_traffic.sort_values(ascending=False).head(10)
    
    # Store the result in the dictionary
    top_users_per_application[app] = top_10_users

# Display results for each application
for app, top_users in top_users_per_application.items():
    print(f"\nTop 10 Users for {app}:")
    print(top_users)
    # Example visualization for Social Media
app = 'Social Media'
top_users_social_media = top_users_per_application[app]

plt.figure(figsize=(10, 6))
top_users_social_media.plot(kind='bar', color='skyblue')
plt.title(f"Top 10 Users for {app}")
plt.xlabel("MSISDN/Number")
plt.ylabel("Total Traffic (Bytes)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# List of applications
applications = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']

# Calculate total traffic for each application
app_traffic = {}
for app in applications:
    # Calculate total traffic (DL + UL) for the application
    total_traffic = data[f'{app} DL (Bytes)'].sum() + data[f'{app} UL (Bytes)'].sum()
    app_traffic[app] = total_traffic

# Convert to a DataFrame for sorting and visualization
app_traffic_df = pd.DataFrame.from_dict(app_traffic, orient='index', columns=['Total_Traffic'])
app_traffic_df = app_traffic_df.sort_values(by='Total_Traffic', ascending=False)

# Select the top 3 applications
top_3_apps = app_traffic_df.head(3)

# Plotting
plt.figure(figsize=(10, 6))

# Bar plot
sns.barplot(
    x=top_3_apps.index,
    y=top_3_apps['Total_Traffic'],
    palette="viridis"
)
plt.title("Top 3 Most Used Applications by Total Traffic")
plt.xlabel("Application")
plt.ylabel("Total Traffic (Bytes)")
plt.tight_layout()
plt.show()

# Pie chart
plt.figure(figsize=(8, 8))
top_3_apps['Total_Traffic'].plot(
    kind='pie',
    autopct='%1.1f%%',
    colors=sns.color_palette("viridis", n_colors=3)
)
plt.title("Traffic Distribution of Top 3 Applications")
plt.ylabel("")  # Hide y-label for cleaner pie chart
plt.tight_layout()
plt.show()
# Aggregate engagement metrics per user
customer_metrics = data.groupby('MSISDN/Number').agg({
    'Bearer Id': 'count',  # Session frequency
    'Dur. (ms)': 'sum',    # Total session duration
    'Total DL (Bytes)': 'sum',  # Total download traffic
    'Total UL (Bytes)': 'sum'   # Total upload traffic
}).rename(columns={
    'Bearer Id': 'Session_Frequency',
    'Dur. (ms)': 'Total_Session_Duration',
    'Total DL (Bytes)': 'Total_Download_Bytes',
    'Total UL (Bytes)': 'Total_Upload_Bytes'
})

# Calculate total traffic (download + upload)
customer_metrics['Total_Traffic'] = customer_metrics['Total_Download_Bytes'] + customer_metrics['Total_Upload_Bytes']

# Normalize metrics
scaler = MinMaxScaler()
normalized_metrics = scaler.fit_transform(customer_metrics[['Session_Frequency', 'Total_Session_Duration', 'Total_Traffic']])
# Elbow method to find the optimal k
inertia = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(normalized_metrics)
    inertia.append(kmeans.inertia_)

# Plot inertia vs k
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='--', color='b')
plt.title('Elbow Method to Determine Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.grid()
plt.show()

# Choose the optimal k (from the elbow point) and run k-means clustering
optimal_k = 3  # Example: Replace with your observed elbow point
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(normalized_metrics)

# Add cluster labels to the original DataFrame
customer_metrics['Engagement_Cluster'] = clusters
# Analyze cluster characteristics
cluster_summary = customer_metrics.groupby('Engagement_Cluster').mean()

# Display cluster summary
print("Cluster Summary:")
print(cluster_summary)
#Experience Analytics
# Replace missing values for numerical columns with the mean
numerical_columns = [
    'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
    'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
    'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)'
]
for col in numerical_columns:
    data[col].fillna(data[col].mean(), inplace=True)

# Replace missing values for categorical columns with the mode
categorical_columns = ['Handset Type']
for col in categorical_columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Treat outliers for numerical columns (IQR method)
def treat_outliers(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
    data.loc[outliers, column] = data[column].mean()

for col in numerical_columns:
    treat_outliers(col)
    # Aggregate metrics per customer (MSISDN/Number)
customer_metrics = data.groupby('MSISDN/Number').agg({
    # Average TCP retransmission
    'TCP DL Retrans. Vol (Bytes)': 'mean',
    'TCP UL Retrans. Vol (Bytes)': 'mean',
    # Average RTT
    'Avg RTT DL (ms)': 'mean',
    'Avg RTT UL (ms)': 'mean',
    # Most frequent handset type
    'Handset Type': lambda x: x.mode()[0],
    # Average throughput
    'Avg Bearer TP DL (kbps)': 'mean',
    'Avg Bearer TP UL (kbps)': 'mean'
}).rename(columns={
    'TCP DL Retrans. Vol (Bytes)': 'Avg_TCP_DL_Retrans_Bytes',
    'TCP UL Retrans. Vol (Bytes)': 'Avg_TCP_UL_Retrans_Bytes',
    'Avg RTT DL (ms)': 'Avg_RTT_DL_ms',
    'Avg RTT UL (ms)': 'Avg_RTT_UL_ms',
    'Avg Bearer TP DL (kbps)': 'Avg_Throughput_DL_kbps',
    'Avg Bearer TP UL (kbps)': 'Avg_Throughput_UL_kbps'
})

# Display the aggregated data
print("Aggregated Metrics Per Customer:")
print(customer_metrics.head())
# Handle missing values by replacing with mean
numerical_columns = [
    'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)',
    'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
    'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)'
]
for col in numerical_columns:
    data[col].fillna(data[col].mean(), inplace=True)

# Function to compute top, bottom, and most frequent values
def compute_stats(column):
    top_10 = column.sort_values(ascending=False).head(10).tolist()  # Top 10
    bottom_10 = column[column > 0].sort_values().head(10).tolist()  # Bottom 10 (non-zero)
    most_frequent = column.value_counts().head(10).index.tolist()  # 10 most frequent values
    return top_10, bottom_10, most_frequent

# Compute stats for TCP, RTT, and Throughput
metrics = {
    "TCP DL Retrans": data['TCP DL Retrans. Vol (Bytes)'],
    "TCP UL Retrans": data['TCP UL Retrans. Vol (Bytes)'],
    "RTT DL": data['Avg RTT DL (ms)'],
    "RTT UL": data['Avg RTT UL (ms)'],
    "Throughput DL": data['Avg Bearer TP DL (kbps)'],
    "Throughput UL": data['Avg Bearer TP UL (kbps)']
    }

results = {}
for metric_name, metric_data in metrics.items():
    results[metric_name] = compute_stats(metric_data)

# Display results
for metric_name, (top_10, bottom_10, most_frequent) in results.items():
    print(f"\n{metric_name}:")
    print("Top 10:", top_10)
    print("Bottom 10:", bottom_10)
    print("Most Frequent:", most_frequent)
    print("that is good")
    # Convert 'Avg Bearer TP DL (kbps)' and 'Avg Bearer TP UL (kbps)' to bytes/second for throughput calculations
data['Avg Bearer TP DL (Bps)'] = data['Avg Bearer TP DL (kbps)'] * 125  # 1 kbps = 125 bytes/s
data['Avg Bearer TP UL (Bps)'] = data['Avg Bearer TP UL (kbps)'] * 125

# Calculate total average throughput per handset type
data['Total Throughput (Bps)'] = data['Avg Bearer TP DL (Bps)'] + data['Avg Bearer TP UL (Bps)']

# Group by handset type
handset_group = data.groupby('Handset Type')

# (d) Distribution of average throughput per handset type
avg_throughput = handset_group['Total Throughput (Bps)'].mean().sort_values(ascending=False)

# (e) Average TCP retransmissions per handset type
data['Total TCP Retransmissions (Bytes)'] = data['TCP DL Retrans. Vol (Bytes)'] + data['TCP UL Retrans. Vol (Bytes)']
avg_tcp_retrans = handset_group['Total TCP Retransmissions (Bytes)'].mean().sort_values(ascending=False)

# Display results
print("Distribution of Average Throughput per Handset Type:")
print(avg_throughput)

print("\nAverage TCP Retransmissions per Handset Type:")
print(avg_tcp_retrans)
# Define features (engagement and experience metrics) and target (Satisfaction Score)
features = [
    'Total UL (Bytes)', 'Total DL (Bytes)', 'Activity Duration DL (ms)', 
    'Activity Duration UL (ms)', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 
    'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)'
]
target = 'Satisfaction_Score'

# Convert all feature columns to numeric, coercing non-numeric values to NaN
for col in features:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Fill missing values with column mean
data[features] = data[features].fillna(data[features].mean())

# Ensure Satisfaction Score is calculated
data['Engagement_Score'] = data[['Total UL (Bytes)', 'Total DL (Bytes)', 'Activity Duration DL (ms)', 'Activity Duration UL (ms)']].mean(axis=1)
data['Experience_Score'] = data[['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']].mean(axis=1)
data['Satisfaction_Score'] = (data['Engagement_Score'] + data['Experience_Score']) / 2

# Feature scaling
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Split data into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R^2 Score: {r2}")

# Display feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)
#Save the updated dataset
data.to_csv('Satisfication_score.csv', index=False)
print("\nsatisfication scores assigned and saved as 'Satisfication_score.csv'")

# Define engagement and experience metrics
engagement_metrics = ['Total UL (Bytes)', 'Total DL (Bytes)', 'Activity Duration DL (ms)', 'Activity Duration UL (ms)']
experience_metrics = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']

# Convert all feature columns to numeric, coercing non-numeric values to NaN
for col in engagement_metrics + experience_metrics:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Fill missing values with column means
data[engagement_metrics + experience_metrics] = data[engagement_metrics + experience_metrics].fillna(
    data[engagement_metrics + experience_metrics].mean()
)

# Calculate Engagement Score and Experience Score
data['Engagement_Score'] = data[engagement_metrics].mean(axis=1)
data['Experience_Score'] = data[experience_metrics].mean(axis=1)

# Normalize Engagement and Experience Scores
scaler = MinMaxScaler()
data[['Engagement_Score', 'Experience_Score']] = scaler.fit_transform(data[['Engagement_Score', 'Experience_Score']])

# Apply K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[['Engagement_Score', 'Experience_Score']])

# Extract cluster centroids
centroids = kmeans.cluster_centers_

# Visualize clusters
plt.figure(figsize=(8, 6))
plt.scatter(data['Engagement_Score'], data['Experience_Score'], c=data['Cluster'], cmap='viridis', label='Customers')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title("K-Means Clustering (k=2)")
plt.xlabel("Engagement Score")
plt.ylabel("Experience Score")
plt.legend()
plt.show()

# Display cluster summary
cluster_summary = data.groupby('Cluster')[['Engagement_Score', 'Experience_Score']].mean()
print("\nCluster Summary:")
print(cluster_summary)
# Ensure Satisfaction Score is calculated
data['Satisfaction_Score'] = (data['Engagement_Score'] + data['Experience_Score']) / 2

# Group by clusters and calculate the mean Satisfaction and Experience scores
cluster_summary = data.groupby('Cluster')[['Satisfaction_Score', 'Experience_Score']].mean()

# Display the results
print("\nAverage Satisfaction & Experience Score Per Cluster:")
print(cluster_summary)
import mysql.connector

# Database connection
connection = mysql.connector.connect(
    host='localhost',
    user='root',
    password='hasan',
    database='telecom_data_analysis'
)

cursor = connection.cursor()

# Create table SQL
create_table_query = """
CREATE TABLE IF NOT EXISTS telecom_data_table (
    IMSI INT AUTO_INCREMENT PRIMARY KEY,
    engagement_score INT NOT NULL,
    experience_score INT NOT NULL,
    satisfaction_score INT NOT NULL
);
"""
cursor.execute(create_table_query)

# Commit changes to the database
connection.commit()

# Fetch and display the data
cursor.execute("SELECT * FROM telecom_table")
rows = cursor.fetchall()
for row in rows:
    print(row)

# Close the cursor and connection
cursor.close()
connection.close()

# File path and database connection details
file_path = "C:\\Users\\Hasan\\Desktop\\data science folder\\Copy of Week2_challenge_data_source(CSV).csv"
db_host = 'localhost'
db_user = 'root'
db_password = 'hasan'
db_name = 'telecom_data_analysis'
table_name = 'telecom_data_table'

# Load the CSV file into a DataFrame
data = pd.read_csv(file_path)

# Select only relevant columns
data = data[['IMSI', 'Total UL (Bytes)', 'Total DL (Bytes)', 'Activity Duration DL (ms)', 'Activity Duration UL (ms)', 
             'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']]

# Data cleaning
columns_to_convert = ['Total UL (Bytes)', 'Total DL (Bytes)', 'Activity Duration DL (ms)', 'Activity Duration UL (ms)', 
                      'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']

for col in columns_to_convert:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Fill missing values with column means
data[columns_to_convert] = data[columns_to_convert].fillna(data[columns_to_convert].mean())

# Compute scores
data['Engagement_Score'] = data[['Total UL (Bytes)', 'Total DL (Bytes)', 'Activity Duration DL (ms)', 'Activity Duration UL (ms)']].mean(axis=1)
data['Experience_Score'] = data[['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']].mean(axis=1)
data['Satisfaction_Score'] = (data['Engagement_Score'] + data['Experience_Score']) / 2

# Drop duplicates in IMSI
data = data.drop_duplicates(subset='IMSI', keep='first')

# Keep only necessary columns
data_to_insert = data[['IMSI', 'Engagement_Score', 'Experience_Score', 'Satisfaction_Score']]

# Replace NaN values with None (which will be inserted as NULL in MySQL)
data_to_insert = data_to_insert.applymap(lambda x: None if pd.isna(x) else x)

# Ensure correct data types for insertion
data_to_insert['IMSI'] = data_to_insert['IMSI'].astype('Int64')  # Ensure IMSI is integer type (nullable)
data_to_insert['Engagement_Score'] = data_to_insert['Engagement_Score'].astype('float64')  # Ensure scores are float
data_to_insert['Experience_Score'] = data_to_insert['Experience_Score'].astype('float64')
data_to_insert['Satisfaction_Score'] = data_to_insert['Satisfaction_Score'].astype('float64')

# Establish database connection
def get_db_connection():
    return connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name
    )

# Insert data in chunks with retry mechanism
chunk_size = 100  # Number of rows per chunk
max_retries = 3  # Max retry attempts for each chunk
retry_delay = 5  # Delay between retries in seconds

try:
    conn = get_db_connection()
    with conn.cursor() as cursor:
        query = f"""
        INSERT INTO {table_name} (IMSI, engagement_Score, experience_Score, satisfaction_Score)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
        engagement_Score = VALUES(engagement_Score),
        experience_Score = VALUES(experience_Score),
        satisfaction_Score = VALUES(satisfaction_Score);
        """
        
        # Insert in chunks with retries
        for start in range(0, len(data_to_insert), chunk_size):
            end = start + chunk_size
            chunk = data_to_insert.iloc[start:end]
            attempts = 0
            
            while attempts < max_retries:
                try:
                    cursor.executemany(query, chunk.values.tolist())
                    conn.commit()
                    print(f"Inserted rows {start} to {end}")
                    break  # Successful insertion, break retry loop
                except IntegrityError as e:
                    print(f"IntegrityError occurred: {e}")
                    break  # Skip problematic chunk
                except Error as e:
                    print(f"Database error occurred: {e}")
                    attempts += 1
                    if attempts < max_retries:
                        print(f"Retrying ({attempts}/{max_retries}) after {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print(f"Failed to insert rows {start} to {end} after {max_retries} attempts.")
                        break
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    attempts += 1
                    if attempts < max_retries:
                        print(f"Retrying ({attempts}/{max_retries}) after {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print(f"Failed to insert rows {start} to {end} after {max_retries} attempts.")
                        break
finally:
    conn.close()
#MLflow Setup and Model Tracking
# Define engagement and experience metrics
engagement_metrics = ['Total UL (Bytes)', 'Total DL (Bytes)', 'Activity Duration DL (ms)', 'Activity Duration UL (ms)']
experience_metrics = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']

# Convert metrics to numeric and handle non-numeric values
for col in engagement_metrics + experience_metrics:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Fill missing values with column means
data[engagement_metrics + experience_metrics] = data[engagement_metrics + experience_metrics].fillna(
    data[engagement_metrics + experience_metrics].mean()
)

# Calculate Engagement Score and Experience Score
data['Engagement_Score'] = data[engagement_metrics].mean(axis=1)
data['Experience_Score'] = data[experience_metrics].mean(axis=1)

# Calculate Satisfaction Score
data['Satisfaction_Score'] = (data['Engagement_Score'] + data['Experience_Score']) / 2

# Define features and target
features = ['Engagement_Score', 'Experience_Score']
target = 'Satisfaction_Score'
X = data[features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow Tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Replace with your MLflow tracking server URI
mlflow.set_experiment("Customer Satisfaction Prediction")

with mlflow.start_run() as run:
    # Record start time
    start_time = datetime.now()
    
    # Initialize model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Log model parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log metrics
    mlflow.log_metric("Mean Absolute Error", mae)
    mlflow.log_metric("Mean Squared Error", mse)
    mlflow.log_metric("R2 Score", r2)
    
    # Save and log the model
    model_path = "random_forest_model"
    mlflow.sklearn.log_model(model, model_path)
    
    # Record end time
    end_time = datetime.now()
    
    # Log additional information
    mlflow.set_tag("Code Version", os.getenv("GIT_COMMIT", "v1.0"))
    mlflow.set_tag("Source", "Customer Engagement Analysis")
    mlflow.log_param("Start Time", str(start_time))
    mlflow.log_param("End Time", str(end_time))
    
    # Save metrics and parameters to CSV for reporting
    results = pd.DataFrame({
        "Metric": ["Mean Absolute Error", "Mean Squared Error", "R2 Score"],
        "Value": [mae, mse, r2]
    })
    results.to_csv("tracking_results.csv", index=False)
    mlflow.log_artifact("tracking_results.csv")

print("Model tracking and logging completed.")
#Model Deployment Tracking Using Docker
# Initialize Flask app
app = Flask(__name__)

# Function to train the model
def train_model():
    
    # Define features and target
    engagement_metrics = ['Total UL (Bytes)', 'Total DL (Bytes)', 'Activity Duration DL (ms)', 'Activity Duration UL (ms)']
    experience_metrics = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']

    for col in engagement_metrics + experience_metrics:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data[engagement_metrics + experience_metrics] = data[engagement_metrics + experience_metrics].fillna(
        data[engagement_metrics + experience_metrics].mean()
    )

    data['Engagement_Score'] = data[engagement_metrics].mean(axis=1)
    data['Experience_Score'] = data[experience_metrics].mean(axis=1)
    data['Satisfaction_Score'] = (data['Engagement_Score'] + data['Experience_Score']) / 2

    X = data[['Engagement_Score', 'Experience_Score']]
    y = data['Satisfaction_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()

    # Save model
    joblib.dump(model, "model.joblib")

    # Log metrics
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "R2_Score": r2_score(y_test, y_pred),
        "Training_Time": end_time - start_time
    }

    # Save metrics to file
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    return metrics

# API endpoint for model training
@app.route('/train', methods=['POST'])
def train_endpoint():
    metrics = train_model()
    return jsonify({"status": "success", "metrics": metrics})

# API endpoint to make predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model = joblib.load("model.joblib")
    X = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(X)[0]
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)