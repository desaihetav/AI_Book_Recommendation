from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.title("Book Recommendation System")


books = pd.read_csv('books.csv', sep=';',
                    error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication',
                 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
users = pd.read_csv('users.csv', sep=';',
                    error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv('ratings.csv', sep=';',
                      error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']


counts1 = ratings['userID'].value_counts()
ratings = ratings[ratings['userID'].isin(counts1[counts1 >= 200].index)]
counts = ratings['bookRating'].value_counts()
ratings = ratings[ratings['bookRating'].isin(counts[counts >= 100].index)]


combine_book_rating = pd.merge(ratings, books, on='ISBN')
columns = ['yearOfPublication', 'publisher',
           'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
combine_book_rating = combine_book_rating.drop(columns, axis=1)
combine_book_rating.head()


combine_book_rating = combine_book_rating.dropna(axis=0, subset=['bookTitle'])
book_ratingCount = (combine_book_rating.
                    groupby(by=['bookTitle'])['bookRating'].
                    count().
                    reset_index().
                    rename(columns={'bookRating': 'totalRatingCount'})
                    [['bookTitle', 'totalRatingCount']]
                    )
book_ratingCount.head()


rating_with_totalRatingCount = combine_book_rating.merge(
    book_ratingCount, left_on='bookTitle', right_on='bookTitle', how='left')


popularity_threshold = 50
rating_popular_book = rating_with_totalRatingCount.query(
    'totalRatingCount >= @popularity_threshold')


combined = rating_popular_book.merge(
    users, left_on='userID', right_on='userID', how='left')

us_canada_user_rating = combined[combined['Location'].str.contains(
    "usa|canada")]
us_canada_user_rating = us_canada_user_rating.drop('Age', axis=1)


us_canada_user_rating = us_canada_user_rating.drop_duplicates(
    ['userID', 'bookTitle'])
us_canada_user_rating_pivot = us_canada_user_rating.pivot(
    index='bookTitle', columns='userID', values='bookRating').fillna(0)
us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)


model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(us_canada_user_rating_matrix)

num_index = us_canada_user_rating_pivot.reset_index()


all_books = us_canada_user_rating['bookTitle'].unique().tolist()
all_books.sort()
st.subheader("Select A Book")
selected_book = st.selectbox('', all_books)


query_index = -1
for i in range(len(num_index)):
    if num_index['bookTitle'][i] == selected_book:
        query_index = i
        break


us_canada_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1)


distances, indices = model_knn.kneighbors(
    us_canada_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6)


for i in range(0, len(distances.flatten())):
    if i == 0:
        st.header('Recommendations for {0}:\n\n'.format(
            us_canada_user_rating_pivot.index[query_index]))
    else:
        st.write('{0}: {1}, with distance of {2}:'.format(
            i, us_canada_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))
