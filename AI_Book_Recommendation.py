# from sklearn.neighbors import NearestNeighbors
# from scipy.sparse import csr_matrix
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import streamlit as st

# st.title("Book Recommendation System")


# books = pd.read_csv('books.csv', sep=';',
#                     error_bad_lines=False, encoding="latin-1")
# books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication',
#                  'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
# users = pd.read_csv('users.csv', sep=';',
#                     error_bad_lines=False, encoding="latin-1")
# users.columns = ['userID', 'Location', 'Age']
# ratings = pd.read_csv('ratings.csv', sep=';',
#                       error_bad_lines=False, encoding="latin-1")
# ratings.columns = ['userID', 'ISBN', 'bookRating']


# counts1 = ratings['userID'].value_counts()
# ratings = ratings[ratings['userID'].isin(counts1[counts1 >= 200].index)]
# counts = ratings['bookRating'].value_counts()
# ratings = ratings[ratings['bookRating'].isin(counts[counts >= 100].index)]


# combine_book_rating = pd.merge(ratings, books, on='ISBN')
# columns = ['yearOfPublication', 'publisher',
#            'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
# combine_book_rating = combine_book_rating.drop(columns, axis=1)
# combine_book_rating.head()


# combine_book_rating = combine_book_rating.dropna(axis=0, subset=['bookTitle'])
# book_ratingCount = (combine_book_rating.
#                     groupby(by=['bookTitle'])['bookRating'].
#                     count().
#                     reset_index().
#                     rename(columns={'bookRating': 'totalRatingCount'})
#                     [['bookTitle', 'totalRatingCount']]
#                     )
# book_ratingCount.head()


# rating_with_totalRatingCount = combine_book_rating.merge(
#     book_ratingCount, left_on='bookTitle', right_on='bookTitle', how='left')


# popularity_threshold = 50
# rating_popular_book = rating_with_totalRatingCount.query(
#     'totalRatingCount >= @popularity_threshold')


# combined = rating_popular_book.merge(
#     users, left_on='userID', right_on='userID', how='left')

# us_canada_user_rating = combined[combined['Location'].str.contains(
#     "usa|canada")]
# us_canada_user_rating = us_canada_user_rating.drop('Age', axis=1)


# us_canada_user_rating = us_canada_user_rating.drop_duplicates(
#     ['userID', 'bookTitle'])
# us_canada_user_rating_pivot = us_canada_user_rating.pivot(
#     index='bookTitle', columns='userID', values='bookRating').fillna(0)
# us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)


# model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
# model_knn.fit(us_canada_user_rating_matrix)

# num_index = us_canada_user_rating_pivot.reset_index()


# all_books = us_canada_user_rating['bookTitle'].unique().tolist()
# all_books.sort()
# st.subheader("Select A Book")
# selected_book = st.selectbox('', all_books)


# query_index = -1
# for i in range(len(num_index)):
#     if num_index['bookTitle'][i] == selected_book:
#         query_index = i
#         break


# us_canada_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1)


# distances, indices = model_knn.kneighbors(
#     us_canada_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6)


# for i in range(0, len(distances.flatten())):
#     if i == 0:
#         st.header('Recommendations for {0}:\n\n'.format(
#             us_canada_user_rating_pivot.index[query_index]))
#     else:
#         st.write('{0}: {1}, with distance of {2}:'.format(
#             i, us_canada_user_rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))


# st.markdown('''
# <html>
# <hr/>
# </html>
# ''', unsafe_allow_html=True)

# st.header("Artificial Intelligence Mini Project – Semester 7")
# st.subheader("Submitted To:")
# st.write("Prof. Mahesh Maurya")
# st.subheader("Performed By:")
# st.write("Ishita Biswas, B014")
# st.write("Ritik Bochiwal, B015")
# st.write("Hetav Desai, B021")
# st.write("Sara Dharadhar, B024")


import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

def create(stock_symbol, stock_info, document_id, data):
    ''' Function to Create a new row of predicted stock price
    '''
    db.collection(stock_symbol).document(document_id).set(data)
    # db.collection('stock').document(stock_symbol)

def merge(stock_symbol, document_id, data):
    ''' Function to merge new data with a new data paramter to an existing row in a document
    '''
    db.collection('stock').document(document_id).set({'open':'15'}, merge = True)

def read(stock_symbol, stock_info, document_id):
    ''' Function to read a single row from Cloud Firestore
    '''
    # Getting a document with a known ID
    return db.collection('stock').collection(stock_info).document(document_id).get()
    

def remove_row(stock_symbol, document_id):
    ''' Function to remove an entire company's stock prediction row
    '''
    db.collection('stock').document(document_id).delete()

def remove_field(stock_symbol, document_id):
    ''' Functino to remove a single field of a specific company's stock prediction row
    '''
def remove_all_documents():
    ''' Function removes all the documents in the collection
    '''
    docs = db.collection('stock').get()
    for doc in docs:
        key = doc.id  
        db.collection('stock').document(key).delete()

def main():

    # Create and upload data for Cipla
    stock_symbol_1 = 'cipla' # placeholder
    prediction_date_1 = '17-03-2021' # placeholder value for the date which's Close price is predicted by the
    # ML model
    predicted_close_1 = 20 # placeholder value to; the number that the ML model churns out
    document_id_1 = 'cipla_17032021' # (stock_symbol + _ + ddmmyy)
    actual_open_1 = 12 # placeholder
    stock_info_1 = 'cipla_stock_info' # placeholder (stock_symbol + _ + 'stock_info')
    data_1 = {
        'open': actual_open_1, 
        'close': predicted_close_1,
        'prediction_for': prediction_date_1} # placeholder values for now
    create(stock_symbol_1, stock_info_1, document_id_1, data_1)

    # Create and upload data for Tata
    stock_symbol_2 = 'tata' # placeholder
    prediction_date_2 = '17-03-2021' # placeholder value for the date which's Close price is predicted by the
    # ML model; Use YYYY-MM-DD instead
    predicted_close_2 = 87 # placeholder value to; the number that the ML model churns out
    document_id_2 = 'tata_17032021' # (stock_symbol + _ + ddmmyy)
    actual_open_2 = 33 # placeholder
    stock_info_2 = 'tata_stock_info' # placeholder (stock_symbol + _ + 'stock_info')
    label = 'March 17, 21' # Derive from datetime, turn into string (MM(in words) + ' ' + DD + ', ' + YY)
    data_2 = {
        'open': actual_open_2, 
        'close': predicted_close_2,
        'prediction_for': prediction_date_2} # placeholder values for now
    create(stock_symbol_2, stock_info_2, document_id_2, data_2)
    
    
    # Read Cipla data
    result = read(stock_symbol_1, stock_info_1, document_id_1)
    if result.exists:
        print(result.to_dict())

if __name__ == '__main__':
    main()


# (done) TODO : Convert all the CRUD codelines into separate functions
# TODO : Decide on a stock prediction script
# TODO : Make the stock prediction script output the following: Predicted Close price of next day, 
# Predicted Close price of next 5 days, Predicted Close price of next 10 days
# TODO : Make and Save new models of 3-5 different NIFTY-50 companies and use those models to 
# make predictions instead of training a model everytime someone makes a request for prediction
# TODO : You could store the model in the firestore and then retreive it from the firestore with 
# a query to use it every time you need to make prediction
# TODO : Save each new predicted value with the ID: '<stock_symbol>_ddmmyy'. For example, 
# predited stock price of Cipla for 17th March 2021 would be 'cipla_170321'
