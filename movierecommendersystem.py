# ============================================================
# CONTENT-BASED MOVIE RECOMMENDER SYSTEM (VS CODE VERSION)
# ============================================================

# ------------------------------
# 1. Import Required Libraries
# ------------------------------
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------
# 2. Load Dataset
# NOTE: Replace this path with your file location in VS Code
# Example: df = pd.read_csv("TMDB 10000 Movies Dataset.csv")
# ------------------------------
df = pd.read_csv("TMDB 10000 Movies Dataset.csv")


# ------------------------------
# 3. Preprocessing
# Fill missing text fields
# ------------------------------
df['overview'] = df['overview'].fillna('')


# ------------------------------
# 4. TF-IDF Vectorizer
# Convert overview text into numeric vectors
# ------------------------------
tfidf = TfidfVectorizer(stop_words='english')

# Fit & transform
tfidf_matrix = tfidf.fit_transform(df['overview'])


# ------------------------------
# 5. Cosine Similarity
# Measures similarity between movie vectors
# ------------------------------
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# ------------------------------
# 6. Create index mapping (title -> index)
# Helps in finding movie quickly
# ------------------------------
indices = pd.Series(df.index, index=df['title']).drop_duplicates()


# ------------------------------
# 7. Recommender Function
# Input: movie title
# Output: list of similar movie titles
# ------------------------------
def recommend_movie(title, num_recommendations=10):
    if title not in indices:
        return ["Movie not found in dataset."]

    # Get index of selected movie
    idx = indices[title]

    # Get similarity scores for all movies
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies by similarity
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Skip first (same movie), return top recommendations
    sim_scores = sim_scores[1:num_recommendations+1]

    # Extract movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return titles
    return df['title'].iloc[movie_indices].tolist()


# ------------------------------
# 8. Example Usage
# Run program in VS Code terminal:
# python recommender.py
# ------------------------------
if __name__ == "__main__":
    movie_name = input("Enter movie name: ")
    results = recommend_movie(movie_name)

    print("\nRecommended Movies:")
    for movie in results:
        print("-", movie)