import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Page Configuration
st.set_page_config(page_title="Tourism Experience Analytics", layout="wide")

# ------------------------------------------------
# LOAD DATA & MODELS
# ------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("df_master.csv")
    df["Rating"] = pd.to_numeric(df["Rating"], errors='coerce').fillna(0)
    return df

@st.cache_resource
def load_models():
    # Load your trained model and preprocessing files
    visit_model = joblib.load("visit_mode_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return visit_model, scaler

df = load_data()
visit_model, scaler = load_models()

# Mapping numeric visit modes to categorical labels
VISIT_MODE_MAP = {
    1: "Solo",
    2: "Couples",
    3: "Family",
    4: "Friends",
    5: "Business"
}

# ------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------
st.sidebar.title("🌍 Tourism Engine")
app_mode = st.sidebar.selectbox("Choose the App Mode", 
                                ["Project Analytics", "Visit Mode Classifier", "Recommendation System"])

# ------------------------------------------------
# MODE 1: PROJECT ANALYTICS
# ------------------------------------------------
if app_mode == "Project Analytics":
    st.title("📊 Tourism Data Analytics")
    
    selected_city = st.selectbox("Select City", ["All"] + sorted(df["AttractionCityName"].unique().tolist()))
    filtered_df = df[df["AttractionCityName"] == selected_city] if selected_city != "All" else df.copy()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ⭐ Popular Attractions")
        top_attr = filtered_df.groupby("Attraction")["Rating"].count().sort_values(ascending=False).head(10).reset_index()
        fig1 = px.bar(top_attr, x="Rating", y="Attraction", orientation='h', color="Rating", color_continuous_scale="Blues")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("### 👥 Visitor Segmentation")
        # Map values for the chart labels
        pie_df = filtered_df.copy()
        pie_df["VisitModeLabel"] = pie_df["VisitMode"].map(VISIT_MODE_MAP).fillna("Unknown")
        visit_dist = pie_df["VisitModeLabel"].value_counts().reset_index()
        fig2 = px.pie(visit_dist, values="count", names="VisitModeLabel", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------
# MODE 2: VISIT MODE CLASSIFIER (Categorical Fix)
# ------------------------------------------------
elif app_mode == "Visit Mode Classifier":
    st.title("🤖 Traveler Behavior Classifier")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            visit_year = st.slider("Visit Year", 2010, 2025, 2022)
            visit_month = st.slider("Visit Month", 1, 12, 10)
            continent_id = st.number_input("Continent ID", value=5)
        with col2:
            rating = st.slider("Anticipated Rating", 1.0, 5.0, 4.5)
            type_id = st.number_input("Attraction Type ID", value=63)
            # Add other necessary feature defaults based on your scaler
            user_visits = st.number_input("User Total Visits", value=5)

        submitted = st.form_submit_button("Predict Visit Mode")
        
        if submitted:
            # 1. Prepare features (Must match the 12 features your scaler expects)
            # VisitYear, VisitMonth, ContinentId, UserRegionId, UserCountryId, UserCityId, 
            # AttractionTypeId, Attraction_Avg_Rating, Attraction_Total_Visits, 
            # User_Avg_Rating, User_Total_Visits, Rating
            input_data = np.array([[visit_year, visit_month, continent_id, 21.0, 163.0, 4341.0, 
                                    type_id, 4.2, 100, 4.5, user_visits, rating]])
            
            # 2. Scale and Predict
            scaled_input = scaler.transform(input_data)
            prediction_id = visit_model.predict(scaled_input)[0]
            
            # 3. Convert ID to Text
            category_name = VISIT_MODE_MAP.get(prediction_id, "Other/Unknown")
            st.success(f"The Predicted Visit Mode is: **{category_name}**")

# ------------------------------------------------
# MODE 3: RECOMMENDATION SYSTEM (Memory Error Fix)
# ------------------------------------------------
elif app_mode == "Recommendation System":
    st.title("📍 Personalized Recommendations")
    user_id_input = st.number_input("Enter User ID", value=70456)

    if st.button("Generate Recommendations"):
        # Create categories for matrix indexing
        user_cat = df['UserId'].astype("category")
        item_cat = df['AttractionId'].astype("category")
        
        # Build Sparse Matrix (Efficient)
        interaction_matrix = csr_matrix((df['Rating'], (user_cat.cat.codes, item_cat.cat.codes)))
        
        try:
            # Find internal index for the requested User ID
            user_idx = list(user_cat.cat.categories).index(user_id_input)
            
            # MEMORY FIX: Calculate similarity only for THIS user against others
            user_vector = interaction_matrix[user_idx]
            similarities = cosine_similarity(user_vector, interaction_matrix).flatten()
            
            # Get top 5 similar users (excluding self)
            similar_users_indices = similarities.argsort()[-6:-1][::-1]
            
            recommendations = []
            for idx in similar_users_indices:
                sim_user_id = user_cat.cat.categories[idx]
                # Find attractions this similar user liked
                sim_user_data = df[df['UserId'] == sim_user_id][['Attraction', 'AttractionCityName', 'Rating']]
                recommendations.append(sim_user_data)
            
            res_df = pd.concat(recommendations).drop_duplicates(subset=['Attraction']).head(5)
            st.write(f"Based on users similar to {user_id_input}, you might like:")
            st.table(res_df)
            
        except ValueError:
            st.error("User ID not found. Showing top-rated global attractions instead.")
            top_global = df.groupby("Attraction")[["Rating", "AttractionCityName"]].mean().sort_values(by="Rating", ascending=False).head(5)
            st.table(top_global)