import streamlit as st
import pandas as pd
import numpy as np
from src.user_user_cf import UserUserCF

# --- Load Data ---
@st.cache_data
def load_data():
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings_base = pd.read_csv('data/ml-100k/ub.base', sep='\t', names=r_cols)
    ratings_base[['user_id', 'movie_id']] -= 1  # 0-based indexing

    movie_cols = ['movie_id', 'title']
    movies = pd.read_csv('data/ml-100k/u.item', sep='|', encoding='latin-1',
                         usecols=[0, 1], names=movie_cols, header=None)
    return ratings_base.to_numpy(), movies

rate_train, movies_df = load_data()
cf_model = UserUserCF(rate_train, k=30)
cf_model.fit()

# --- Streamlit UI ---
st.title("🎬 Movie Recommender System")

tab1, tab2 = st.tabs(["🔎 Recommend for User", "🎞️ Find Similar Movies"])

with tab1:
    st.header("📌 Get Movie Recommendations")
    user_id = st.number_input("Enter user ID:", min_value=0, max_value=cf_model.n_users - 1, step=1)
    n_recs = st.slider("Number of recommendations:", 1, 20, 10)

    if st.button("Get Recommendations"):
        rec_ids = cf_model.recommend(user_id, n_recommendations=n_recs)
        st.subheader("🎯 Recommended Movies:")
        for idx, movie_id in enumerate(rec_ids):
            title = movies_df[movies_df['movie_id'] == movie_id + 1]['title'].values
            if len(title) > 0:
                st.write(f"{idx+1}. {title[0]}")
            else:
                st.write(f"{idx+1}. [Movie ID {movie_id}]")

with tab2:
    st.header("🎥 Find Movies Similar to a Given Title")
    search_title = st.text_input("Enter a movie title:")

    if st.button("Find Similar Movies"):
        matched = movies_df[movies_df['title'].str.lower().str.contains(search_title.lower())]
        if matched.empty:
            st.warning("No movie found with that name.")
        else:
            selected_movie_id = matched.iloc[0]['movie_id'] - 1  # convert to 0-based
            st.write(f"✅ Matched movie: **{matched.iloc[0]['title']}**")
            # Fake a "user" who only rated this one movie highly
            pseudo_user_data = np.array([[cf_model.n_users, selected_movie_id, 5]])
            cf_model.add(pseudo_user_data)
            cf_model.refresh()
            similar_ids = cf_model.recommend(cf_model.n_users, n_recommendations=10)

            st.subheader("🎬 Similar Movies:")
            for i, movie_id in enumerate(similar_ids):
                title = movies_df[movies_df['movie_id'] == movie_id + 1]['title'].values
                if len(title) > 0:
                    st.write(f"{i+1}. {title[0]}")
