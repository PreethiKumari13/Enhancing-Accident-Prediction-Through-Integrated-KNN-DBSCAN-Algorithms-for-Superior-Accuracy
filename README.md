# 🚗 Road Accident Severity Prediction using KNN & DBSCAN

A hybrid Machine Learning project that combines **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** with **K-Nearest Neighbors (KNN)** and **Random Forest** to improve road accident severity prediction by removing noisy and inconsistent data before classification.

> 🎓 Final Year Project | 📄 Published Research Paper | 🐍 Python | 🤖 Machine Learning | ☁️ Google Colab

---

# 📌 Project Overview

Road traffic accidents are one of the leading causes of injuries and fatalities worldwide. Predicting accident severity can help emergency services, traffic authorities, and intelligent transportation systems respond more effectively.

This project presents a **hybrid machine learning approach** that combines:

- **DBSCAN** for removing noisy and outlier data
- **KNN** for accident severity classification
- **Random Forest** for performance comparison

The complete project was developed and evaluated using **Google Colab**.

---

# 🎯 Problem Statement

Traditional accident prediction models often suffer from:

- Noisy and inconsistent datasets
- Poor handling of outliers
- Reduced prediction accuracy

This project addresses these challenges by integrating **unsupervised learning** with **supervised machine learning** to improve prediction performance.

---

# 💡 Proposed Solution

The project follows the workflow below:

1. Dataset Generation & Preprocessing
2. Feature Scaling
3. Noise Removal using DBSCAN
4. Accident Severity Prediction using KNN
5. Performance Comparison using Random Forest
6. Model Evaluation

---

# ⚙️ Project Workflow

Dataset

⬇️

Data Preprocessing

⬇️

Feature Scaling

⬇️

DBSCAN (Noise Removal)

⬇️

Train-Test Split

⬇️

KNN Model

⬇️

Random Forest Model

⬇️

Performance Evaluation

---

# ✨ Features

- Hybrid Machine Learning Model
- Noise & Outlier Removal using DBSCAN
- Accident Severity Prediction
- KNN Classification
- Random Forest Comparison
- Data Visualization
- Confusion Matrix
- Classification Report
- Scalable Machine Learning Pipeline

---

# 📂 Dataset Description

**Total Records:** 8000

### Input Features

- Speed
- Brake Force
- Traffic Density
- Weather
- Visibility
- Vehicle Weight
- Vehicle Age
- Reaction Time
- Nearby Vehicle Count
- Road Condition
- Lane Count
- Road Curvature
- Time of Day
- Alcohol Level
- Tyre Condition

### Target Variable

**Accident Severity**

- 0 → Low
- 1 → Moderate
- 2 → High

---

# 🛠️ Technologies Used

- Python
- Google Colab
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

---

# 🤖 Machine Learning Models

### DBSCAN

Used to detect and remove noisy data points before classification.

### K-Nearest Neighbors (KNN)

Used as the primary classification algorithm for accident severity prediction.

### Random Forest

Used for comparison and performance evaluation.

---

# 📊 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

The hybrid DBSCAN + KNN approach achieved excellent performance on the generated dataset after removing noisy samples.

---

# 📸 Project Outputs

The project generates:

- Dataset Statistics
- Feature Distribution Graphs
- Box Plots
- Scatter Plots
- Correlation Heatmap
- Confusion Matrix
- KNN Accuracy Comparison
- Classification Reports

---

# 📁 Project Structure

```
Road-Accident-Prediction-KNN-DBSCAN
│
├── accident_prediction.ipynb
├── accident_prediction.py
├── README.md
├── requirements.txt
├── dataset/
│   └── perfect_accident_dataset_final.csv
├── models/
│   ├── knn_accident_model.pkl
│   └── scaler.pkl
├── images/
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   └── accuracy_graph.png
├── Project_Report.pdf
├── Project_Presentation.pdf
└── Published_Paper.pdf
```

---

# ▶️ How to Run

Clone the repository

```bash
git clone https://github.com/PreethiKumari13/Road-Accident-Prediction-KNN-DBSCAN.git
```

Navigate to the project folder

```bash
cd Road-Accident-Prediction-KNN-DBSCAN
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python accident_prediction.py
```

Or open the notebook using Google Colab.

---

# ☁️ Google Colab

Open the notebook directly:

**https://colab.research.google.com/drive/1_l8wKBaP-dkJFLe5X2Ueg6zaygiEODR_?usp=sharing**

---

# 📄 Research Publication

This project is based on our published research paper:

**"Enhancing Accident Prediction Through Integrated KNN & DBSCAN Algorithms for Superior Accuracy"**

The published paper is included in this repository.

---

# 🚀 Future Enhancements

- Real-time traffic data integration
- GPS-based accident hotspot prediction
- Deep Learning (CNN/LSTM)
- Web Application Deployment
- Mobile Application
- Integration with Emergency Response Systems

---

# 👩‍💻 Contributors

- **Ankitha K V**
- **Preethi Kumari**
- **Likhitha M N**

---

# ⭐ Acknowledgement

This project was developed as part of the Bachelor of Engineering (Information Science & Engineering) Final Year Project and demonstrates the application of Machine Learning techniques for intelligent transportation systems.

---

## 📬 Contact

**Preethi Kumari**

GitHub: https://github.com/PreethiKumari13

---

⭐ If you found this project useful, consider giving it a **Star** on GitHub.
