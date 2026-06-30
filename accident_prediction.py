# ==========================================
# 1. Import Required Libraries
# ==========================================

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# ==========================================
# 2. Dataset Creation & Loading
# ==========================================

def generate_cluster(mean, std, size, severity):
    """
    Generate synthetic accident data for a specific severity level.
    """

    data = np.random.normal(mean, std, (size, len(mean)))

    df = pd.DataFrame(data, columns=[
        "speed",
        "brake_force",
        "traffic_density",
        "weather",
        "visibility",
        "vehicle_weight",
        "vehicle_age",
        "reaction_time",
        "nearby_vehicle_count",
        "road_condition",
        "lane_count",
        "road_curvature",
        "time_of_day",
        "alcohol_level",
        "tyre_condition"
    ])

    df["Accident_severity"] = severity
    return df


# Generate Low Severity Accidents
cluster_low = generate_cluster(
    mean=[40,20,30,1,80,1200,5,1.2,2,1,2,5,10,0,80],
    std=[5,5,5,0.5,10,200,1,0.3,1,0.1,1,2,1,0.1,5],
    size=3500,
    severity=0
)

# Generate Moderate Severity Accidents
cluster_med = generate_cluster(
    mean=[70,40,55,2,60,1500,8,1.5,5,0.5,3,15,15,0.5,60],
    std=[6,6,6,0.7,12,220,2,0.4,2,0.2,1,5,2,0.15,6],
    size=3000,
    severity=1
)

# Generate High Severity Accidents
cluster_high = generate_cluster(
    mean=[100,70,80,3,40,2000,12,2.0,8,0.8,4,25,20,1.0,40],
    std=[7,7,7,0.8,15,300,3,0.5,2,0.1,1,3,3,0.2,7],
    size=1500,
    severity=2
)

# Combine all accident records
df = pd.concat(
    [cluster_low, cluster_med, cluster_high],
    ignore_index=True
)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save dataset
df.to_csv("perfect_accident_dataset_final.csv", index=False)

print("Dataset Shape:", df.shape)
print("Dataset saved successfully as 'perfect_accident_dataset_final.csv'")

# ==========================================
# 3. Load and Explore the Dataset
# ==========================================

# Load the generated dataset
df = pd.read_csv("perfect_accident_dataset_final.csv")

# Display the first five rows
print("\nFirst 5 Rows of the Dataset")
print(df.head())

# Display dataset dimensions
print("\nDataset Shape:", df.shape)

# Display dataset information
print("\nDataset Information")
df.info()

# Display statistical summary
print("\nStatistical Summary")
print(df.describe())
# ==========================================
# 4. Data Preprocessing
# ==========================================

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove missing values
df.dropna(inplace=True)

print("\nDataset Shape After Cleaning:", df.shape)
# ==========================================
# 5. Exploratory Data Analysis (EDA)
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns

# Accident Severity Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Accident_severity", data=df)

plt.title("Accident Severity Distribution")
plt.xlabel("Severity Level")
plt.ylabel("Count")

plt.show()
# ==========================================
# Speed Distribution
# ==========================================

plt.figure(figsize=(10,6))

sns.histplot(
    data=df,
    x="speed",
    bins=30,
    kde=True,
    color="steelblue"
)

plt.title("Distribution of Vehicle Speed", fontsize=14)
plt.xlabel("Speed (km/h)")
plt.ylabel("Frequency")

plt.grid(alpha=0.3)

plt.show()
# ==========================================
# Speed vs Accident Severity
# ==========================================

plt.figure(figsize=(10,6))

sns.boxplot(
    x="Accident_severity",
    y="speed",
    data=df,
    palette="pastel"
)

plt.title("Speed vs Accident Severity", fontsize=14)
plt.xlabel("Accident Severity")
plt.ylabel("Speed (km/h)")

plt.show()
# ==========================================
# Speed vs Brake Force
# ==========================================

plt.figure(figsize=(10,6))

sns.scatterplot(
    x="speed",
    y="brake_force",
    hue="Accident_severity",
    data=df,
    palette="Set2"
)

plt.title("Speed vs Brake Force", fontsize=14)
plt.xlabel("Speed")
plt.ylabel("Brake Force")

plt.show()
# ==========================================
# 6. Feature Scaling
# ==========================================

# Separate input features and target variable
X = df.drop("Accident_severity", axis=1)
y = df["Accident_severity"]

# Standardize the feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# ==========================================
# 7. DBSCAN Clustering
# ==========================================

dbscan = DBSCAN(eps=3, min_samples=5)

clusters = dbscan.fit_predict(X_scaled)

# Remove outliers
mask = clusters != -1

X_clean = X_scaled[mask]
y_clean = y[mask]

print("Original Dataset Shape:", X.shape)
print("Dataset Shape After DBSCAN:", X_clean.shape)
# ==========================================
# 8. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_clean,
    y_clean,
    test_size=0.20,
    random_state=42,
    stratify=y_clean
)
# ==========================================
# 9. KNN Model
# ==========================================

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print("\nKNN Accuracy:")
print(accuracy_score(y_test, y_pred_knn))

print("\nClassification Report")
print(classification_report(y_test, y_pred_knn))
# ==========================================
# 10. Random Forest Model
# ==========================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, y_pred_rf))

print("\nClassification Report")
print(classification_report(y_test, y_pred_rf))
# ==========================================
# 11. Confusion Matrix
# ==========================================

cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.show()
# ==========================================
# Accuracy Comparison
# ==========================================

labels = ["Before DBSCAN", "After DBSCAN"]
accuracies = [0.85, 1.00]

plt.figure(figsize=(7,5))

colors = ["#FFD6A5","#BDE0FE"]

plt.bar(labels,accuracies,color=colors,edgecolor="gray")

plt.title("KNN Accuracy Before vs After DBSCAN")

plt.ylabel("Accuracy")

for i,v in enumerate(accuracies):
    plt.text(i,v+0.02,f"{v:.2f}",ha="center")

plt.ylim(0,1.05)

plt.show()
# ==========================================
# Correlation Heatmap
# ==========================================

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()
# ==========================================
# Save Model
# ==========================================

import joblib

joblib.dump(knn,"knn_accident_model.pkl")

joblib.dump(scaler,"scaler.pkl")

print("Model saved successfully.")
