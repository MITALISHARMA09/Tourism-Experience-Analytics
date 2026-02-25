# 🌍 Tourism Experience Analytics
Classification, Prediction & Recommendation System
# 📌 Overview

Tourism Experience Analytics is an end-to-end Machine Learning project designed to enhance user experience in the tourism domain.

This system performs:

📊 Regression – Predict attraction ratings

🏷️ Classification – Predict visit mode (Business, Family, Couples, etc.)

🎯 Recommendation – Personalized attraction suggestions

The final solution is deployed using Streamlit as an interactive web application.

# 🎯 Problem Statement

Tourism agencies and travel platforms aim to:

Provide personalized recommendations

Predict user satisfaction levels

Classify user travel behavior

Increase customer retention

This project leverages user demographics, travel history, and attraction features to build predictive and recommendation models.

# 💼 Business Use Cases

✔ Personalized attraction recommendations

✔ Tourism trend analytics

✔ Customer segmentation

✔ Targeted marketing campaigns

✔ Improved customer satisfaction & retention

# 🛠️ Tech Stack

Python

Pandas, NumPy

Matplotlib, Seaborn, Plotly

Scikit-learn

LightGBM / Random Forest / XGBoost

Collaborative Filtering (Cosine Similarity)

SQL

Streamlit

# 📂 Dataset Description

The project uses a structured Tourism dataset consisting of:

1️⃣ Transaction Data

TransactionId

UserId

VisitYear

VisitMonth

VisitMode

AttractionId

Rating

2️⃣ User Data

UserId

ContinentId

RegionId

CountryId

CityId

3️⃣ Attraction (Item) Data

AttractionId

AttractionCityId

AttractionTypeId

Attraction

AttractionAddress

4️⃣ Supporting Tables

City Data

Country Data

Region Data

Continent Data

Visit Mode Data

Attraction Type Data

# 🚀 Project Objectives
🔹 1. Regression – Predict Attraction Ratings

Goal: Predict the rating (1–5 scale) a user would give to an attraction.

Evaluation Metrics:

R² Score

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

🔹 2. Classification – Predict Visit Mode

Goal: Predict whether a user will travel for:

Business

Family

Couples

Friends

Solo

Evaluation Metrics:

Accuracy

Precision

Recall

F1 Score

🔹 3. Recommendation System

Approaches Used:

🔁 Collaborative Filtering

📌 Content-Based Filtering

🔀 Hybrid Recommendation (Optional)

Output: Ranked list of personalized attractions.

# 🔍 Project Workflow
1️⃣ Data Preparation

Handling missing values

Removing duplicates

Resolving inconsistent categorical values

Outlier detection

Encoding categorical variables

Feature scaling

2️⃣ Exploratory Data Analysis (EDA)

User distribution across continents & countries

Popular attraction types

Rating distribution analysis

VisitMode vs Demographics correlation

Regional tourism trends

3️⃣ Feature Engineering

Aggregated user-level features

Merging multiple relational tables

Creating user-item interaction matrix

Label encoding

4️⃣ Model Training

Linear Regression

Random Forest

LightGBM / XGBoost

Collaborative Filtering using Cosine Similarity

5️⃣ Model Evaluation

Cross-validation

Model comparison

Business-focused interpretation

6️⃣ Deployment

A Streamlit Web Application that allows users to:

Input demographic details

Get predicted visit mode

View predicted attraction rating

Receive personalized recommendations

Explore tourism trends through visual dashboards

# 📁 Project Structure
Tourism-Experience-Analytics/
│
├── data/
│   ├── raw_data.csv
│   ├── cleaned_data.csv
│
├── notebooks/
│   ├─ data_cleaning.ipynb
│   ├─ tourism_experience_analytics.ipynb

│
├── models/
│   ├── visit_mode_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│
├── app.py
└── README.md
# ▶️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/Tourism-Experience-Analytics.git
cd Tourism-Experience-Analytics
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Streamlit App
streamlit run app.py

# 👩‍💻 Author

Mitali Sharma
B.Tech – Computer Science
Machine Learning & Data Analytics Enthusiast

# 🌟 Future Improvements

Deep learning-based recommendation

Real-time feedback integration

Cloud deployment (AWS/GCP)

Location-based smart recommendations
