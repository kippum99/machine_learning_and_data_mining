import numpy as np

# movie metadata encoded as indicated in MOVIE_FEATURES

MOVIE_FEATURES = ["Movie Id", "Movie Title", "Unknown", "Action", "Adventure", "Animation", "Childrens", "Comedy", "Crime",
         "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
         "Thriller", "War", "Western"]

def parse(filename = 'data/movies.txt'):
    genresdict = {}
    genres = MOVIE_FEATURES[3:]
    for g in genres:
        genresdict[g] = []

    for row in np.loadtxt(filename, dtype=str, delimiter='\t', encoding='latin1'):
        movie_id = int(row[0])
        movie_genres = row[3:]
        for index, indicator in enumerate(movie_genres):
            if indicator=="1":
                genre = genres[index]
                genresdict[genre].append(movie_id)
    return genresdict


def load_ratings(filename='data/data.txt'):
    return np.loadtxt(filename, dtype=int, delimiter='\t')

def popular_movies(ratings, n=10):
    max_index = len(set(ratings[:,1]))+1
    movie_hist = np.histogram(ratings[:,1], bins=range(1, max_index+1))[0]
    return [id for _, id in sorted(zip(movie_hist, range(1, max_index)), reverse=True)[:n]]

def best_movies(ratings, n=10):
    max_index = len(set(ratings[:,1]))+1
    avg_ratings = [np.mean(ratings[np.where(ratings[:,1] == i),2]) for i in range(1, max_index)]
    return [id for _, id in sorted(zip(avg_ratings, range(1, max_index)), reverse=True)[:n]]
