import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_file
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import os

app = Flask(__name__)

# Load datasets
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    books_path = os.path.join(base_dir, 'data', 'Final_data.csv')
    ratings_path = os.path.join(base_dir, 'data', 'Ratings.csv')

    final_books = pd.read_csv(books_path, on_bad_lines='skip', encoding='latin1')
    final_books.fillna('', inplace=True)
    final_books.rename(columns={'ï»¿author': 'author'}, inplace=True)

    ratings = pd.read_csv(ratings_path, on_bad_lines='skip', encoding='latin1')
    final_books.columns = final_books.columns.str.strip()
    ratings.columns = ratings.columns.str.strip()
    final_books.rename(columns={'ï»¿book_Id': 'book_id'}, inplace=True)
    
    return final_books, ratings

final_books, ratings = load_data()

# Content-based filtering setup
final_books['combined_features'] = final_books['title'] + ' ' + final_books['genre'] + ' ' + final_books['author'].fillna('') + ' ' + final_books['desc']
count_vectorizer = CountVectorizer(stop_words='english', max_features=5000)
count_matrix = count_vectorizer.fit_transform(final_books['combined_features'])
tfidf_transformer = TfidfTransformer()
tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)

def compute_cosine_similarity(tfidf_matrix, idx, top_n=10):
    cosine_similarities = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[-top_n-1:-1][::-1]
    similar_scores = cosine_similarities[similar_indices]
    return similar_indices, similar_scores

def get_recommendations(book_id, top_n=10):
    idx = final_books[final_books['book_id'] == book_id].index[0]
    similar_indices, similar_scores = compute_cosine_similarity(tfidf_matrix, idx, top_n)
    return final_books.iloc[similar_indices][['book_id', 'isbn13']].to_dict(orient='records')

# Collaborative filtering setup
def train_collaborative_model():
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    model = SVD(n_factors=20, biased=True, random_state=42)
    model.fit(trainset)
    return model

model = train_collaborative_model()

def get_book_recommendations(user_id, model, books_df, top_n=10):
    all_book_ids = books_df['book_id'].unique()
    predictions = [model.predict(user_id, book_id) for book_id in all_book_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_book_ids = [pred.iid for pred in predictions[:top_n]]
    recommended_books = books_df[books_df['book_id'].isin(top_book_ids)]
    return recommended_books[['book_id', 'isbn13']].to_dict(orient='records')

@app.route('/content-based-recommendation', methods=['GET'])
def content_based_recommendation():
    book_id = int(request.args.get('book_id'))
    recommendations = get_recommendations(book_id)
    return jsonify(recommendations)

@app.route('/collaborative-recommendation', methods=['GET'])
def collaborative_recommendation():
    user_id = int(request.args.get('user_id'))
    recommendations = get_book_recommendations(user_id, model, final_books)
    return jsonify(recommendations)

@app.route('/update-ratings', methods=['POST'])
def update_ratings():
    global ratings, model
    ratings = pd.read_csv(os.path.join('data', 'Ratings.csv'), on_bad_lines='skip', encoding='latin1')
    ratings.columns = ratings.columns.str.strip()
    model = train_collaborative_model()
    return jsonify({"message": "Ratings data updated and model retrained."})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
