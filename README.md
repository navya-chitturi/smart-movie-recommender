# Smart Movie Recommendation System

This is a mini project that recommends movies to users based on their preferences using Machine Learning techniques.

## Objective
To suggest top 5 movie recommendations using:
-  Collaborative Filtering (based on user ratings)
-  Content-Based Filtering (based on genres)

## Tools & Technologies
- Python
- Pandas
- Scikit-learn
- Streamlit
- MovieLens 100k Dataset

---

## Workflow Summary

1. **Load Dataset**  
   - MovieLens 100k dataset (`u.data`, `u.item`) loaded using Pandas

2. **Data Preprocessing**  
   - Merged user ratings with movie metadata  
   - Cleaned and saved as `cleaned_movies.csv`

3. **Collaborative Filtering**  
   - Created a user-movie rating matrix  
   - Calculated movie-to-movie correlation using Pearson method

4. **Content-Based Filtering**  
   - Used movie genres to create genre vectors  
   - Calculated cosine similarity between movies based on genres

5. **Web Interface**  
   - Developed an interactive Streamlit app (`app.py`)  
   - User selects a movie and recommendation method  
   - App returns top 5 movie suggestions

---

## Project Structure

| File/Folder | Description |
|-------------|-------------|
| `app.py` | Streamlit web app file |
| `movie_recommender.ipynb` | Notebook for data preprocessing and model building |
| `cleaned_movies.csv` | Cleaned dataset used in the app |
| `u.data` | Original ratings dataset |
| `u.item` | Original movie metadata |
| `README.md` | Project overview and instructions |

---

## How to Run the App Locally

1. Install required libraries:
   ```bash
   pip install pandas scikit-learn streamlit
   
2. Run the Streamlit app
Open your terminal or Anaconda Prompt, navigate to the project folder, and run:
    ```bash
   streamlit run app.py

3. View the app in your browser
After a few seconds, the app will open automatically at:

   http://localhost:8501

---
🔗 **Live Demo**: [Click here to try the app](https://smart-movie-recommender-gwwfj69ryvc3b26zvtzrr5.streamlit.app)

🔗 **Live Demo**: [Click here to try the app](https://your-final-streamlit-link)
