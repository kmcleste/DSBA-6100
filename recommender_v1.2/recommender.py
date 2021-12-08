from networkx.algorithms.distance_measures import radius
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import re
import operator
import os
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import json
import ast

# set app threads to 16 (max of my system)
os.environ["NUMEXPR_MAX_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
pd.options.mode.chained_assignment = None
# force matplotlib graphs to be same shape
plt.rcParams.update({"figure.autolayout": True})
# set the overall layout to a wide format to fit the network graph on a tradition 1920x1080 screen
st.set_page_config(layout='wide', page_title='BlockCluster')


def main():
    # Create streamlit web app
    st.markdown("# BlockCluster v1.2")
    st.write("---")
    rec_options = st.sidebar.expander("Recommendation Options", expanded=True)
    data_options = st.sidebar.expander("View Dataset", expanded=False)
    option_1 = rec_options.checkbox("Genre")
    option_2 = rec_options.checkbox("Title")
    option_3 = rec_options.checkbox("IMDB Weighted Ratings")
    option_4 = rec_options.checkbox("Compare User Ratings")
    option_5 = rec_options.checkbox("Score User Similarity")
    option_6 = rec_options.checkbox("Predict User Rating")
    option_7 = rec_options.checkbox("Collaborative User Recommendations")
    option_8 = data_options.checkbox("Show Full Dataset")
    option_9 = data_options.checkbox("Show User Movies")
    option_10 = data_options.checkbox("Visualizations")
    option_11 = data_options.checkbox("User Network Graph (BETA)")

    st.markdown(
        "## Welcome!\nPlease select the options by which you would like to receive recommendations in the sidebar\n\nFor more detailed instructions, see the dropdown below"
    )

    # Short descriptions for the various options in the st.sidebar, exapandable
    expand = st.expander("More information...", expanded=False)
    expand.markdown(
          "- **Genre**: Returns movies with similar genres to the user input\n"
        + "- **Title**: Returns movies with similar titles to the user input\n"
        + "- **IMDB Weighted Ratings**: Returns top/bottom X movies using a weighted rating\n"
        + "- **Compare User Ratings**: Returns list of movies watched by both users and their ratings\n"
        + "- **Score User Similarity**: Returns the similarity of 2 users using Euclidean distance\n"
        + "- **Predict User Rating**: Returns the predicted rating for a movie title using collaborative filtering\n"
        + "- **Collaborative User Recommendations**: Returns list of recommendations using collaborative filtering\n"
        + "- **Show Full Dataset**: Returns all data in MovieLens dataset\n"
        + "- **Show User Movies**: Returns list of movies viewed by selected user\n"
        + "- **Visualizations**: Draws basic bar graph showing the number of movies produced per year\n"
        + "- **User Network Graph**: Renders an interactive network graph showing user/movie relationships"
    )

    st.write('---')

    # load in the data
    movies = pd.read_csv("data/movielens/100k/movies.csv", sep=",", encoding="utf-8")
    ratings = pd.read_csv("data/movielens/100k/ratings.csv", sep=",", encoding="utf-8", usecols=["userId", "movieId", "rating"],)

    # remove rows with null values

    # split genres into string array
    movies["genres"] = movies["genres"].str.split("|")
    movies["genres"] = movies["genres"].fillna("").astype("str")

    # extract year from title column
    movies["year"] = [re.search("(\d){4}", x) for x in movies["title"]]
    movies = movies.dropna()
    movies["year"] = [re.search("(\d){4}", x).group() for x in movies["title"]]
    movies = movies.replace(to_replace="\((.*)\)", value="", regex=True)
    movies = movies.replace(to_replace=" +$", value="", regex=True)
    movies["year"] = movies["year"].astype("int")

    # remove title that were missing a year or are recorded after this study
    movies = movies[movies["year"] <= 2018]

    merged_df = ratings.join(movies.set_index("movieId"), on="movieId")
    merged_df = merged_df.dropna(axis=0)
    merged_df["year"] = [int(x) for x in merged_df["year"]]
    merged_df = merged_df[merged_df["year"] >= 1915]

    with open('profiles.json') as json_file:
        profiles = json.load(json_file)

    @st.cache(show_spinner=False)
    def euclidean_distance(points):
        squared_diff = [(point[0] - point[1]) ** 2 for point in points]
        summed_squared_diffs = sum(squared_diff)
        distance = np.sqrt(summed_squared_diffs)
        return distance

    # converts euclidean distance to a similarity measure
    @st.cache(show_spinner=False)
    def similarity(reviews):
        return 1 / (1 + euclidean_distance(reviews))

    @st.cache(show_spinner=False)
    def get_user_similarity(user_A, user_B):
        reviews = get_reviews(user_A, user_B)
        return similarity(reviews)

    @st.cache(show_spinner=False)
    def generate_tfidf(column):
        tf = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 4), min_df=0, stop_words="english"
        )

        tfidf_matrix = tf.fit_transform(movies[column])

        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        titles = movies["title"]
        indices = pd.Series(movies.index, index=movies["title"])
        return cosine_sim, titles, indices

    @st.cache(show_spinner=False)
    def genre_recommendations(title, num_titles):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1 : num_titles + 1]
        movie_indices = [i[0] for i in sim_scores]
        return titles.iloc[movie_indices]

    @st.cache(show_spinner=False)
    def get_common_movies(user_A, user_B):
        return [movie for movie in profiles[user_A] if movie in profiles[user_B]]

    @st.cache(show_spinner=False)
    # get reviews from common movies
    def get_reviews(user_A, user_B):
        common_movies = get_common_movies(user_A, user_B)
        return [
            (profiles[user_A][movie], profiles[user_B][movie])
            for movie in common_movies
        ]

    @st.cache(show_spinner=False)
    def recommend_movies(user, num_suggestions, profiles):
        similarity_scores = [
            (get_user_similarity(user, other), other)
            for other in profiles
            if other != user
        ]

        # get sim scores for all users
        similarity_scores.sort()
        similarity_scores.reverse()

        recommendations = {}
        for similarity, other in similarity_scores:
            reviewed = profiles[other]
            for movie in reviewed:
                if movie not in profiles[user]:
                    weight = similarity * reviewed[movie]
                    if movie in recommendations:
                        sim, weights = recommendations[movie]
                        recommendations[movie] = (sim + similarity, weights + [weight])
                    else:
                        recommendations[movie] = (similarity, [weight])

        # PREDICT USER RECOMMENDATION
        for recommendation in recommendations:
            similarity, movie = recommendations[recommendation]
            recommendations[recommendation] = sum(movie) / similarity

        # print(recommendations['Let It Be Me'])

        sorted_recommendations = sorted(
            recommendations.items(), key=operator.itemgetter(1), reverse=True
        )
        df = pd.DataFrame(columns=["Title", "Rating"])
        sorted_recommendations = sorted_recommendations[:num_suggestions]
        titles = []
        ratings = []
        for i, _ in enumerate(sorted_recommendations):
            titles.append(sorted_recommendations[i][0])
            ratings.append(f"{sorted_recommendations[i][1]:.1f}")
        df["Title"] = titles
        df["Rating"] = ratings
        return df

    @st.cache(show_spinner=False)
    # get reviews from common movies
    def get_reviews_detailed(user_A, user_B):
        common_movies = get_common_movies(user_A, user_B)
        reviews = [
            (profiles[user_A][movie], profiles[user_B][movie])
            for movie in common_movies
        ]
        result = pd.DataFrame(columns=["Movie"])
        result["Movie"] = common_movies
        user_A = []
        user_B = []
        for review in reviews:
            user_A.append(f"{review[0]:.1f}")
            user_B.append(f"{review[1]:.1f}")
        result["user_A"] = user_A
        result["user_B"] = user_B
        return result

    @st.cache(show_spinner=False)
    def convert_df(df):
        return df.to_csv().encode("utf-8")

    def generate_plotly(merged_df):
        year = merged_df["year"].unique()
        counts = merged_df["year"].value_counts()
        fig = px.bar(merged_df, year, counts)
        fig.update_layout(
            title="Total Movies Produced by Year",
            xaxis_title="Year",
            yaxis_title="Movie's Produced",
        )
        return fig

    def predict_rating(user, title, profiles):
        similarity_scores = [(get_user_similarity(user, other), other) for other in profiles if other != user]
        # get sim scores for all users
        similarity_scores.sort()
        similarity_scores.reverse()
        
        recommendations = {}
        for similarity, other in similarity_scores:
            reviewed = profiles[other]
            for movie in reviewed:
                weight = similarity * reviewed[movie]
                if movie in recommendations:
                    sim, weights = recommendations[movie]
                    recommendations[movie] = (sim + similarity, weights + [weight])
                else:
                    recommendations[movie] = (similarity, [weight])
                        
        for recommendation in recommendations:
            similarity, movie = recommendations[recommendation]
            recommendations[recommendation] = sum(movie) / similarity

        return recommendations[title]

    def generate_ratings_plot(ratings):
        rating_options = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        rating_count = [ratings[ratings['rating']==x]['rating'].count() for x in rating_options]
        rating_counts = pd.DataFrame(columns=['rating','count'])
        rating_counts['rating'] = rating_options
        rating_counts['count'] = rating_count
        fig = px.bar(data_frame=rating_counts, x='rating', y='count', width=800, height=400, title='Ratings Distribution')
        return fig

    def generate_genre_plot(movies):
        genres = []
        z=[ast.literal_eval(x) for x in movies['genres']]
        for i in z:
            for j in i:
                genres.append(j)
        genres = pd.DataFrame(data=genres)
        x = pd.unique(genres[0])
        y = [genres.value_counts()[x] for x in range(0,len(genres.value_counts()))]
        xy = pd.DataFrame(columns=['genre','count'])
        xy['genre'] = x
        xy['count'] = y
        xy[xy['genre']==''] = 'null'
        fig = px.bar(data_frame=xy, x='genre', y='count', width=800, height=400, title='Genres Distribution')
        return fig

    csv = convert_df(merged_df)
    st.sidebar.download_button(
        label="Download Dataset", data=csv, file_name="movie-dataset.csv", mime="text/csv", key="download-csv"
    )

    weighted_ratings = pd.read_csv('weighted_ratings.csv',sep=',',encoding='utf-8',header=0)

    # genre based recommendations using tf-idf and cosine sim
    if option_1:
        # genre based recommendations
        cosine_sim, titles, indices = generate_tfidf("genres")
        with st.form("genre_selection"):
            st.markdown("### Genre-based Recommendations")
            input_col1, input_col2 = st.columns(2)
            title = input_col1.selectbox("Movie Title", merged_df["title"].unique())
            num_recommendations = input_col2.number_input(
                "Number of Recommendations", min_value=1, max_value=50, value=10, step=1
            )
            submitted = st.form_submit_button("Find Movies")
            if submitted:
                try:
                    st.write(genre_recommendations(title, num_recommendations))
                except KeyError:
                    st.warning("Please enter a valid title")
                except ValueError:
                    st.warning("Something went wrong...try a different title")

    # title based recommendations using tf-idf and cosine sim
    if option_2:
        # genre based recommendations
        cosine_sim, titles, indices = generate_tfidf("title")
        with st.form("title_selection"):
            st.markdown("### Title-based Recommendations")
            input_col1, input_col2 = st.columns(2)
            title = input_col1.selectbox("Movie Title", merged_df["title"].unique())
            num_recommendations = input_col2.number_input(
                "Number of Recommendations", min_value=1, max_value=50, value=10, step=1
            )
            submitted = st.form_submit_button("Find Movies")
            if submitted:
                try:
                    st.write(genre_recommendations(title, num_recommendations))
                except KeyError:
                    st.warning("Please enter a valid title")
                except ValueError:
                    st.warning("Something went wrong...try a different title")

    # return imdb style weighted ratings
    if option_3:
        with st.form("imdb-ratings"):
            st.markdown('### IMDB Weighted Rating')
            col1, col2 = st.columns(2)
            order = col1.selectbox('Order',('Low->High','High->Low'))
            num_recommendations = col2.number_input("Number of Movies", min_value=1, max_value=50, value=10, step=1)
            submitted = st.form_submit_button("Show Movies")
            if submitted and order=='Low->High':
                st.write(weighted_ratings.sort_values(by='score', ascending=True).head(num_recommendations))
            if submitted and order=='High->Low':
                st.write(weighted_ratings.sort_values(by='score', ascending=False).head(num_recommendations))

    # compare user viewed movies and ratings
    if option_4:
        with st.form("compare_user_ratings"):
            st.markdown("### Compare User Ratings")
            input_col1, input_col2 = st.columns(2)
            user_A = input_col1.text_input("User A", "user_1")
            user_B = input_col2.text_input("User B", "user_610")
            submitted = st.form_submit_button("Show Ratings")
            if submitted:
                try:
                    df = get_reviews_detailed(user_A, user_B)
                    st.dataframe(df)
                except KeyError:
                    st.warning("Enter a valid user")

    # show the similarity of 2 user profiles using euclidean distance
    if option_5:
        with st.form("get_user_sim"):
            st.markdown("### User Similarity")
            input_col1, input_col2 = st.columns(2)
            user_A = input_col1.text_input("User A", "user_1")
            user_B = input_col2.text_input("User B", "user_610")
            submitted = st.form_submit_button("Show Similarity")
            if submitted:
                try:
                    st.write(get_user_similarity(user_A, user_B))
                except KeyError:
                    st.warning("Enter a valid user")

    # predict user rating
    if option_6:
        with st.form("predict-rating"):
            st.markdown("### Predict User Rating")
            col1, col2 = st.columns(2)
            user = col1.text_input('User', 'user_1')
            title = col2.selectbox("Movie Title", merged_df["title"].unique())
            submitted = st.form_submit_button("Predict Rating")
            if submitted:
                st.text(f'Predicted Rating: {predict_rating(user, title, profiles):.3f}')
                try:
                    st.text(f'Actual Rating: {profiles[user][title]}')
                except KeyError:
                    st.text('User has not rated this movie.')

    # collaborative filtering, return list of recommendations for given user based on similar users most liked movies
    if option_7:
        with st.form("rec_movie_user"):
            st.markdown("### User-based Recommendations")
            input_col1, input_col2 = st.columns(2)
            user = input_col1.text_input("User", "user_1")
            num_recommendations = input_col2.number_input(
                "Number of Recommendations", min_value=1, max_value=50, value=10
            )
            submitted = st.form_submit_button("Show Recommendations")
            if submitted:
                try:
                    st.dataframe(recommend_movies(user, num_recommendations, profiles))
                except KeyError:
                    st.warning("Enter a valid user")

    # render the full movielens dataset
    if option_8:
        st.markdown('### Full MovieLens Dataset')
        st.write(merged_df)

    # search for movies viewed by user input
    if option_9:
        with st.form("user_movies"):
            st.markdown("### User-Viewed Movies")
            user = st.number_input(
                "User ID", min_value=1, max_value=609, value=1, step=1
            )
            submitted = st.form_submit_button("Show Movies")
            if submitted:
                try:
                    temp_df = merged_df[merged_df["userId"] == user]
                    temp_df["year"] = [int(x) for x in temp_df["year"]]
                    temp_df["rating"] = [(f"{x:.1f}") for x in temp_df["rating"]]
                    st.dataframe(temp_df)
                except KeyError:
                    st.warning("Enter a valid user")

    # generate plot for movies produced by year and clustering charts
    if option_10:
        fig0 = generate_plotly(merged_df)
        st.plotly_chart(fig0, use_container_width=True)
        col1, col2 = st.columns(2)
        fig1 = generate_ratings_plot(ratings)
        fig2 = generate_genre_plot(movies)
        
        col1.plotly_chart(fig1, use_container_width=True)
        col2.plotly_chart(fig2, use_container_width=True)

    # render pyvis/networx network graph for 7 users since system cannot handle any more...
    if option_11:
        st.markdown("### Network Graph")
        HtmlFile = open("network_graph.html", "r", encoding="utf-8")
        source_code = HtmlFile.read()
        components.html(source_code, width=900, height=900)


if __name__ == "__main__":
    main()
