A simple and effective Content-Based Movie Recommender System built using Python, TF-IDF, and Cosine Similarity.
This project uses the TMDB 10,000 Movies Dataset to recommend similar movies based on their overview/description.

# Features

Content-Based filtering

Recommends movies using TF-IDF vectorization

Cosine similarity for nearest movie match

Clean, simple Python code

Works in VS Code / PyCharm / Jupyter

Beginner-friendly ML project

# Requirements

Install the dependencies:

pip install pandas scikit-learn

 Project Structure
Movie-Recommender-System
│
├── movierecommendersystem.py     # Main code
├── TMDB 10000 Movies Dataset.csv # Dataset
└── README.md                     # Project documentation

# How It Works

Load the movie dataset

Extract text features from the overview column

Convert text to TF-IDF vectors

Compute movie similarities using cosine similarity

Recommend movies with the highest similarity score

 # Run the Project

Open terminal:

python movierecommendersystem.py


Enter a movie name:

Enter movie name: The Godfather


Output:

Recommended Movies:
- The Godfather Part II
- Casino
- Once Upon a Time in America
...

# Example Code (Main Logic)
def recommend_movie(title, num_recommendations=10):
    if title not in indices:
        return ["Movie not found in dataset."]

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# Dataset Information

10,000 Movies

Columns:

title

overview

popularity

vote_average
etc.


#Future Improvements

Add genre-based recommendation

## Contribute

Feel free to fork this repo, make changes, and submit a PR.

# License

This project is licensed under the MIT License.
