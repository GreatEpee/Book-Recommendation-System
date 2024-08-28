import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

# Data loading
books_df = pd.read_csv("Data/Books.csv", low_memory=False)
ratings_df = pd.read_csv("Data/Ratings.csv", low_memory=False).sample(40000)
users_df = pd.read_csv("Data/Users.csv", low_memory=False)

# check for 'User-ID' column
if 'User-ID' not in ratings_df.columns or 'User-ID' not in users_df.columns:
    print("Error: 'User-ID' column not found in one or both DataFrames.")
    exit()

users_ratings_df = ratings_df.merge(users_df, on='User-ID')
books_users_ratings = books_df.merge(users_ratings_df, on='ISBN')
books_users_ratings = books_users_ratings[['ISBN', 'User-ID', 'Book-Title', 'Book-Author', 'Book-Rating']]
books_users_ratings.reset_index(drop=True, inplace=True)

# Creating unique ID for books
unique_ids = books_users_ratings['ISBN'].unique()
isbn_to_id = {isbn: i for i, isbn in enumerate(unique_ids)}
books_users_ratings['unique_id_book'] = books_users_ratings['ISBN'].map(isbn_to_id)

# Creating user-book matrix
users_books_pivot_matrix_df = books_users_ratings.pivot(index='User-ID', columns='unique_id_book', values='Book-Rating').fillna(0)

# Convert to matrix for SVD
users_books_pivot_matrix = users_books_pivot_matrix_df.values

# Perform SVD
NUMBER_OF_FACTORS_MF = 15
U, sigma, Vt = svds(users_books_pivot_matrix, k=NUMBER_OF_FACTORS_MF)
sigma = np.diag(sigma)

# Predict ratings
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)

# Cosine Similarity Function
def top_cosine_similarity(data, book_id, top_n=10):
    book_row = data[book_id, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(book_row, data.T) / (magnitude[book_id] * magnitude)
    sort_index = np.argsort(-similarity)
    return sort_index[:top_n]

# Recommendation Function
def similar_books(book_user_rating, book_id, top_indexes):
    print(f'Recommendations for {book_user_rating[book_user_rating.unique_id_book == book_id]["Book-Title"].values[0]}:\n')
    for idx in top_indexes:
        title = book_user_rating[book_user_rating.unique_id_book == idx]['Book-Title'].values[0]
        print(title)

# Example: Finding similar books
k = 50
book_id = 11486
top_n = 5
sliced_Vt = Vt.T  # uses all latent factors
top_indexes = top_cosine_similarity(sliced_Vt, book_id, top_n)
similar_books(books_users_ratings, book_id, top_indexes)
