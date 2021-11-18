from operator import sub
import string
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit as st
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import regularizers
from tensorflow.keras import metrics
from tensorflow.keras.utils import plot_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def main():
    # Create streamlit web app
    st.title('Movie Recommender Systems')

    # 3 Different Recommenders
    # Simple recommenders: offer a generalized recommendation to every user based on movie popularity/genre
    # Content based recommenders: suggest similar items based on a particular item
    # Collaborative filtering: try to predict rating preference

    # Create simple recommender using IMDB weighted rating: wr = (vount_count/(vote_count+negative_vote))*vote_average + (negative_vote/(vote_count+negative_vote))*average_vote

    # load metadata into dataframe
    # dtypes = {
    #     "adult" : object,
    #     "belongs_to_collection": object,
    #     "budget" : float,
    #     "genres": object,
    #     "homepage": object,
    #     "id": int,
    #     "imdb_id": object,
    #     "original_language": object,
    #     "original_title": object,
    #     "overview": object,
    #     "popularity": float,
    #     "poster_path": object,
    #     "production_companies": object,
    #     "production_countries": object,
    #     "release_date": int,
    #     "revenue": float,
    #     "runtime": int,
    #     "spoken_languages": object,
    #     "status": object,
    #     "tagline": object,
    #     "title": object,
    #     "video": object,
    #     "vote_average": float,
    #     "vote_count": int
    # }
    metadata = pd.read_csv('data/movies_metadata.csv')
    # print head to streamlit
    # st.write(metadata.head())
    # st.write(f'Metadata entries: {len(metadata)}')
    # st.write(f'Columns: \n{metadata.columns}')

    # We are going to filter out all movies not in the top 10%
    C = metadata['vote_average'].mean()
    # # st.write('C value: {:.3f}'.format(C))
    m = metadata['vote_count'].quantile(0.90)
    # # st.write('m value: {}'.format(m))

    # # Create a copy of dataframe for filtering
    q_movies = metadata.copy().loc[metadata['vote_count'] >= m]
    # # st.write('Shape of Qualified Movies: {}'.format(q_movies.shape))

    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        # Calculation based on the IMDB formula
        return (v/(v+m) * R) + (m/(m+v) * C)

    # # Define a new feature 'score' and calculate its value with `weighted_rating()`
    q_movies['score'] = q_movies.apply(weighted_rating, axis= 1)

    # # Sort movies based on score calculated above
    q_movies.sort_values(by= 'score', ascending= False, inplace= True)
    st.write('Top 20 Movies by Rating')
    st.write(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(20))

    # Create content-based recommendation system
    st.write('Content-Based Recommender System')
    # st.write(metadata['overview'].head())

    # Check null values
    # st.write(f'null values: {metadata["overview"].isnull().sum()}')
    metadata['overview'] = metadata['overview'].fillna('')
    # st.write(f'null values: {metadata["overview"].isnull().sum()}')

    # Extract features; compute similarity; compute TF-IDF vectors for each document
    tfidf = TfidfVectorizer(stop_words= 'english')
    tfidf_matrix = tfidf.fit_transform(metadata['overview'])
    # st.write(f'tfdif shape: {tfidf_matrix.shape}')
    # st.write('sample of words used to describe movies: {}'.format(tfidf.get_feature_names()[7000:7010]))

    # Use cosine similarity to compute a similarity score
    # Calculating the dot product between each vector will directly return the cosine similarity score, so we will use sklearn linear_kernel()

    tfidf_matrix = tfidf_matrix[:30000] # 30000 for prevent allocate more memory than is available. It has restarted.
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    # st.write(f'Cosine Similarity Shape: {cosine_sim.shape}') # Each movie will be a 1*30000 column vector where each column will be its similarity to each movie
    indices = pd.Series(index=metadata['title'], data = metadata.index).drop_duplicates()

    # Function that takes in movie title as input and outputs most similar movies
    def get_recommendations(title, cosine_sim= cosine_sim):
        # Get the index of the movie that matches the title
        idx = indices[title]
        
        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key= lambda x : x[1], reverse= True)
        
        # Get the scores of the 10 most similar movies, we stared with 1 because index 0 will give the same name of the input title 
        sim_scores = sim_scores[1:11]
        
        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return the top 10 most similar movies
        return metadata['title'][movie_indices]


    # Load keywords and credits
    credits = pd.read_csv('data/credits.csv')
    keywords = pd.read_csv('data/keywords.csv')

    # Remove rows with bad IDs.
    metadata = metadata.drop([19730, 29503, 35587])

    # Convert IDs to int. Required for merging
    keywords['id'] = keywords['id'].astype('int')
    credits['id'] = credits['id'].astype('int')
    metadata['id'] = metadata['id'].astype('int')

    # Merge keywords and credits into your main metadata dataframe
    metadata = metadata.merge(credits, on='id')
    metadata = metadata.merge(keywords, on='id')
    # st.write(metadata.head())

    # st.write(metadata.columns)
    # st.write(len(metadata))

    # Limit the data for memory size
    metadata = metadata[:20000]

    features = ['cast', 'crew', 'keywords', 'genres']
    # st.write(metadata[features].head())

    for feature in features:
        metadata[feature] = metadata[feature].apply(literal_eval)

    # st.write(metadata[features].head())

    def get_director(x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

    def get_list_or_top_3(x):
        if isinstance(x, list):
            names = [i['name'] for i in x]
            # get just 3 or less
            if len(names) > 3:
                names = names[:3]
            return names

        # return empty list if it missing data!
        return []

    # Define new director
    metadata['director'] = metadata['crew'].apply(get_director)

    # Define new director cast, genres and keywords features.
    features = ['cast', 'keywords', 'genres']
    for feature in features:
        metadata[feature] = metadata[feature].apply(get_list_or_top_3)
        
    # Function to convert all strings to lower case and strip names of spaces
    def clean_text(txt):
        if isinstance(txt, list):
            return [i.replace(" ", "").lower() for i in txt]
        else:
            #Check if director exists. If not, return empty string
            if isinstance(txt, str):
                return txt.replace(" ", "").lower()
            else:
                return ''

    # Apply clean_data function to your features.
    features = ['cast', 'keywords', 'director', 'genres']

    for feature in features:
        metadata[feature] = metadata[feature].apply(clean_text)

    def create_soup(x):
        return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

    # Create a new soup feature
    metadata['soup'] = metadata.apply(create_soup, axis=1)

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(metadata['soup'])

    # count_matrix = count_matrix[:30000] # 20000 for prevent allocate more memory than is available. It has restarted.
    cosine_sim_2 = cosine_similarity(count_matrix, count_matrix)

    # Reset index of our main DataFrame and construct reverse mapping as before
    metadata = metadata.reset_index()
    indices = pd.Series(index=metadata['title'], data = metadata.index )
    indices = pd.Series(index=metadata['title'], data = metadata.index).drop_duplicates()


    with st.form("movie_selection"):
        input = st.text_input('Movie Title', 'Enter title...', help='Type in the movie you would like to find similar matches to')

        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write(get_recommendations(input, cosine_sim_2))






if __name__ == "__main__":
    main()
