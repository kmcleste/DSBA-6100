from networkx.algorithms.distance_measures import radius
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import re
import operator
import os
import numpy as np
from pathlib import Path
import plotly.express as go
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from streamlit.state.session_state import Value
import matplotlib.pyplot as plt
import streamlit.components.v1 as components


os.environ["NUMEXPR_MAX_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
pd.options.mode.chained_assignment = None
plt.rcParams.update({"figure.autolayout": True})
st.set_page_config(layout="wide")


def main():
    # Create streamlit web app
    st.markdown("# BlockCluster v1.1")
    st.write("---")
    rec_options = st.sidebar.expander("Recommendation Options", expanded=True)
    data_options = st.sidebar.expander("View Dataset", expanded=False)
    option_1 = rec_options.checkbox("Genre")
    option_2 = rec_options.checkbox("Title")
    option_3 = rec_options.checkbox("Top/Bottom")
    option_4 = rec_options.checkbox("Compare Viewed Movies")
    option_5 = rec_options.checkbox("Compare User Ratings")
    option_6 = rec_options.checkbox("Score User Similarity")
    option_7 = rec_options.checkbox("Collaborative User Recommendations")
    option_8 = data_options.checkbox("Show List of Movies")
    option_9 = data_options.checkbox("Show User Movies")
    option_10 = data_options.checkbox("Visualizations")
    option_11 = data_options.checkbox("User Network Graph (BETA)")

    st.markdown(
        "## Welcome!\nPlease select the options by which you would like to receive recommendations in the sidebar\n\nFor more detailed instructions, see the dropdown below"
    )
    expand = st.expander("More information...", expanded=False)
    expand.markdown(
        "- **Genre**: Returns movies with similar genres to the user input\n"
        + "- **Title**: Returns movies with similar titles to the user input\n"
        + "- **Top/Bottom**: Returns the top or bottom n-movies in the dataset (can take up to a minute to run)\n"
        + "- **Compare Viewed Movies**: Returns movies watched by both user_A and user_B\n"
        + "- **Compare User Ratings**: Returns list of movies watched by both users and their ratings\n"
        + "- **Score User Similarity**: Returns the similarity of 2 users using Euclidean distance\n"
        + "- **Recommend Movie by User**: Returns list of recommendations based on similar user profiles\n"
        + "- **Show List of Movies**: Returns list of all movies in the MovieLens-100k dataset\n"
        + "- **Show User Movies**: Returns list of movies viewed by a given user"
    )

    links = pd.read_csv("data/movielens/100k/links.csv", sep=",", encoding="latin-1")
    movies = pd.read_csv("data/movielens/100k/movies.csv", sep=",", encoding="latin-1")
    ratings = pd.read_csv(
        "data/movielens/100k/ratings.csv",
        sep=",",
        encoding="latin-1",
        usecols=["userId", "movieId", "rating"],
    )
    # tags = pd.read_csv('data/movielens/100k/tags.csv', sep=',', encoding='latin-1', usecols=['userId','movieId','tag'])

    # remove rows with null values
    links = links.dropna()
    links.isnull().sum()

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

    @st.cache(show_spinner=False)
    def euclidean_distance(points):
        squared_diff = [(point[0] - point[1]) ** 2 for point in points]
        summed_squared_diffs = sum(squared_diff)
        distance = np.sqrt(summed_squared_diffs)
        return distance

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
    def generate_profiles(merged_df):
        profiles = {}
        for user in merged_df["userId"].unique():
            username = "user_" + str(user)

            temp = merged_df[merged_df["userId"] == user]

            titles = temp["title"]
            ratings = temp["rating"]
            ratings = list(ratings)

            movie_list = {}
            for i, title in enumerate(titles):
                movie_list[title] = ratings[i]

            profiles[username] = movie_list
        return profiles

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
        # similarity_scores = similarity_scores[0:num_suggestions]

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

        for recommendation in recommendations:
            similarity, movie = recommendations[recommendation]
            recommendations[recommendation] = sum(movie) / similarity

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

    profiles = generate_profiles(merged_df)

    csv = convert_df(merged_df)
    st.sidebar.download_button(
        "Download Dataset", csv, "movie-dataset.csv", "text/csv", key="download-csv"
    )

    def generate_plotly(merged_df):
        year = merged_df["year"].unique()
        counts = merged_df["year"].value_counts()
        fig = go.bar(merged_df, year, counts)
        fig.update_layout(
            title="Total Movies Produced by Year",
            xaxis_title="Year",
            yaxis_title="Movie's Produced",
        )
        return fig

    def generate_elbow(merged_df):
        encoder = LabelEncoder()
        encoded_df = merged_df
        encoded_df["genres_cat"] = encoder.fit_transform(encoded_df["genres"])
        encoded_df["title_cat"] = encoder.fit_transform(encoded_df["title"])
        encoded_df = encoded_df.drop(["title", "genres"], axis=1)
        pca = PCA(n_components=2)
        pca_array = pca.fit_transform(encoded_df)
        sum_of_squares = []
        K = range(1, 10)
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(pca_array)
            sum_of_squares.append(km.inertia_)
        fig = plt.figure(figsize=(7, 7))
        plt.plot(K, sum_of_squares, "bx-")
        plt.xlabel("k")
        plt.ylabel("Sum of Square Distances")
        plt.title("Elbow Method for Optimal k")
        return fig, pca_array

    def generate_kmeans(pca_array):
        kmeans = KMeans(n_clusters=3)
        label = kmeans.fit_predict(pca_array)
        u_labels = np.unique(label)
        fig = plt.figure(figsize=(7, 7))
        for i in u_labels:
            plt.scatter(x=pca_array[label == i, 0], y=pca_array[label == i, 1], label=i)
        plt.legend(u_labels)
        plt.xlabel("Principle Component 1")
        plt.ylabel("Principle Compotent 2")
        plt.title("K-means Clustering")
        return fig

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

    # average rating for each movie
    if option_3:
        user_avg_ratings = pd.DataFrame(columns=["userId", "avg_rating"])
        userId = []
        avg_rating = []
        for user in merged_df["userId"].unique():
            userId.append(user)
            avg_rating.append(
                merged_df[merged_df["userId"] == user]["rating"].sum()
                / len(merged_df[merged_df["userId"] == user])
            )
        user_avg_ratings["userId"] = userId
        user_avg_ratings["avg_rating"] = avg_rating
        user_avg_ratings = user_avg_ratings.set_index("userId")
        avg_movie_ratings = pd.DataFrame(columns=["movieId", "avg_rating"])
        movieId = []
        avg_rating = []
        for movie in merged_df["movieId"].unique():
            movieId.append(movie)
            avg_rating.append(
                merged_df[merged_df["movieId"] == movie]["rating"].sum()
                / len(merged_df[merged_df["movieId"] == movie])
            )
        avg_movie_ratings["movieId"] = movieId
        avg_movie_ratings["avg_rating"] = avg_rating
        avg_movie_ratings = avg_movie_ratings.join(
            movies.set_index("movieId"), on="movieId"
        )
        avg_movie_ratings = avg_movie_ratings.drop(["genres", "year"], axis=1)
        avg_movie_ratings = avg_movie_ratings.dropna()

        with st.form("average_rating"):
            st.markdown("### Average Ratings")
            input_col1, input_col2 = st.columns(2)
            option = input_col1.selectbox("Movie Order", ("Low->High", "High->Low"))
            num_recommendations = input_col2.number_input(
                "Number of Recommendations",
                min_value=1,
                max_value=1000,
                value=10,
                step=1,
            )
            submitted = st.form_submit_button("Show Movies")
            if submitted and option == "Low->High":
                st.write(
                    avg_movie_ratings.sort_values(by=["avg_rating"])[
                        :num_recommendations
                    ]
                )
            if submitted and option == "High->Low":
                st.write(
                    avg_movie_ratings.sort_values(by=["avg_rating"], ascending=False)[
                        :num_recommendations
                    ]
                )

    # user profiles
    if option_4:
        with st.form("compare_users"):
            st.markdown("### Compare User Viewed Movies")
            input_col1, input_col2 = st.columns(2)
            user_A = input_col1.text_input("User A", "user_1")
            user_B = input_col2.text_input("User B", "user_610")
            submitted = st.form_submit_button("Show Common Movies")
            if submitted:
                try:
                    st.write(get_common_movies(user_A, user_B))
                except KeyError:
                    st.warning("Enter a valid user")

    # compare user ratings
    if option_5:
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

    if option_6:
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

    if option_8:
        st.write(merged_df["title"])

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

    if option_10:
        fig0 = generate_plotly(merged_df)
        fig1, pca_array = generate_elbow(merged_df)
        fig2 = generate_kmeans(pca_array)

        st.plotly_chart(fig0, use_container_width=True)

        col1, col2 = st.columns(2)

        col1.pyplot(fig1)
        col2.pyplot(fig2)

    if option_11:
        st.markdown("### Network Graph")
        HtmlFile = open("network_graph.html", "r", encoding="utf-8")
        source_code = HtmlFile.read()
        components.html(source_code, width=1200, height=1200)


if __name__ == "__main__":
    main()
