import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #fef6f0, #fceae8);
        background-attachment: fixed;
        background-size: cover;
        font-family: 'Segoe UI', sans-serif;
        color: #3e3e3e;
    }

    .stButton>button {
        background-color: #f7c8c2;
        color: #3e3e3e;
        border-radius: 15px;
        padding: 8px 20px;
        font-weight: 500;
        border: none;
        box-shadow: 1px 2px 5px rgba(0,0,0,0.1);
    }

    .stButton>button:hover {
        background-color: #f4b6ae;
        color: white;
    }

    h1, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #c96480;
    }

    .stRadio > div {
        background-color: #fff6f4;
        padding: 10px;
        border-radius: 10px;
    }

    .stSelectbox, .stTextInput, .stTextArea {
        background-color: #fff0eb;
        border-radius: 12px;
    }

    .css-1aumxhk, .css-qrbaxs {
        color: #4b4b4b;
    }

    .stMarkdown {
        font-size: 17px;
    }
    </style>
""", unsafe_allow_html=True)



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
