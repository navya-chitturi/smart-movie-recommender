import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the cleaned movie dataset
df = pd.read_csv("cleaned_movies.csv")

# Create user-movie ratings matrix for collaborative filtering
user_movie_ratings = df.pivot_table(index='user_id', columns='title', values='rating')
movie_similarity = user_movie_ratings.corr(method='pearson', min_periods=50)

# Genre similarity setup
genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
              'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movie_genres = df.drop_duplicates(subset='title').set_index('title')[genre_cols]
genre_sim_df = pd.DataFrame(cosine_similarity(movie_genres), index=movie_genres.index, columns=movie_genres.index)

# --- Functions ---
def get_similar_movies_collab(movie_name, rating_given):
    if movie_name not in movie_similarity.columns:
        return ["Movie not found in rating database."]
    similar_scores = movie_similarity[movie_name] * (rating_given - 2.5)
    return similar_scores.sort_values(ascending=False).dropna().head(5).index.tolist()

def recommend_by_genre(title, top_n=5):
    if title not in genre_sim_df.columns:
        return ["Movie not found in genre database."]
    similar_movies = genre_sim_df[title].sort_values(ascending=False).iloc[1:top_n+1]
    return similar_movies.index.tolist()

# --- Streamlit UI ---
st.title("🎬 Smart Movie Recommendation System")

movie_list = sorted(df['title'].unique())
selected_movie = st.selectbox("Choose a movie you like:", movie_list)
method = st.radio("Select Recommendation Method:", ["Collaborative Filtering", "Genre-based Filtering"])

if st.button("Get Recommendations"):
    st.subheader("📽️ Top 5 Recommended Movies")
    if method == "Collaborative Filtering":
        results = get_similar_movies_collab(selected_movie, 5)
    else:
        results = recommend_by_genre(selected_movie)

    for i, movie in enumerate(results, 1):
        st.write(f"{i}. {movie}")
