import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import re
import operator
import os
import numpy as np

from streamlit.state.session_state import Value

os.environ["NUMEXPR_MAX_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
pd.options.mode.chained_assignment = None


def main():
    # Create streamlit web app
    st.markdown("# BlockCluster v1.1")
    st.write("---")
    st.sidebar.subheader("Options")
    option_1 = st.sidebar.checkbox("Genre")
    option_2 = st.sidebar.checkbox("Title")
    option_3 = st.sidebar.checkbox("Top/Bottom")
    option_4 = st.sidebar.checkbox("Compare Viewed Movies")
    option_5 = st.sidebar.checkbox("Compare User Ratings")
    option_6 = st.sidebar.checkbox("Score User Similarity")
    option_7 = st.sidebar.checkbox("Recommend Movie by User")
    option_8 = st.sidebar.checkbox("Show List of Movies")
    option_9 = st.sidebar.checkbox("Show User Movies")
    st.markdown(
        "## Welcome!\nPlease select the options by which you would like to receive recommendations in the sidebar\n"
        + "- Genre: Returns movies with similar genres to the user input\n"
        + "- Title: Returns movies that have similar titles - i.e. Toy Story -> [Toy Soldier, Story of Us]\n"
        + "- Top/Bottom: Returns the top or bottom n-movies in the dataset\n"
        + "- Compare Viewed Movies: Returns movies watched by both user_A and user_B\n"
        + "- Compare User Ratings: Returns list of movies watched by both users and their ratings\n"
        + "- Score User Similarity: Returns similarity of 2 users using Euclidean distance\n"
        + "- Recommend Movie by User: Returns list of recommendations for user_A based on user_B watch/rating history"
    )
    print(os.getcwd())
    links = pd.read_csv("kmcleste/DSBA-6100/blob/main/eda/data/movielens/100k/links.csv", sep=",", encoding="latin-1")
    movies = pd.read_csv("kmcleste/DSBA-6100/blob/main/eda/data/movielens/100k/movies.csv", sep=",", encoding="latin-1")
    ratings = pd.read_csv(
        "kmcleste/DSBA-6100/blob/main/eda/data/movielens/100k/ratings.csv",
        sep=",",
        encoding="latin-1",
        usecols=["userId", "movieId", "rating"],
    )
    tags = pd.read_csv(
        "kmcleste/DSBA-6100/blob/main/eda/data/movielens/100k/tags.csv",
        sep=",",
        encoding="latin-1",
        usecols=["userId", "movieId", "tag"],
    )

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

    def euclidean_distance(points):
        squared_diff = [(point[0] - point[1]) ** 2 for point in points]
        summed_squared_diffs = sum(squared_diff)
        distance = np.sqrt(summed_squared_diffs)
        return distance

    def similarity(reviews):
        return 1 / (1 + euclidean_distance(reviews))

    def get_user_similarity(user_A, user_B):
        reviews = get_reviews(user_A, user_B)
        return similarity(reviews)

    def generate_tfidf():
        tf = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 4), min_df=0, stop_words="english"
        )

        tfidf_matrix = tf.fit_transform(movies["genres"])

        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        titles = movies["title"]
        indices = pd.Series(movies.index, index=movies["title"])
        return cosine_sim, titles, indices

    def genre_recommendations(title, num_titles):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1 : num_titles + 1]
        movie_indices = [i[0] for i in sim_scores]
        return titles.iloc[movie_indices]

    def title_recommendations(title, num_titles):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1 : num_titles + 1]
        movie_indices = [i[0] for i in sim_scores]
        return titles.iloc[movie_indices]

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

    def get_common_movies(user_A, user_B):
        return [movie for movie in profiles[user_A] if movie in profiles[user_B]]

    # get reviews from common movies
    def get_reviews(user_A, user_B):
        common_movies = get_common_movies(user_A, user_B)
        return [
            (profiles[user_A][movie], profiles[user_B][movie])
            for movie in common_movies
        ]

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

    profiles = generate_profiles(merged_df)

    if option_1:
        # genre based recommendations
        cosine_sim, titles, indices = generate_tfidf()
        with st.form("movie_selection"):
            st.markdown("### Genre-based Recommendations")
            input_col1, input_col2 = st.columns(2)
            title = input_col1.text_input(
                "Movie Title",
                "Enter title...",
                help="Type in the movie you would like recommendations based upon",
            )
            num_recommendations = input_col2.number_input(
                "Number of Recommendations", min_value=1, max_value=50, value=10, step=1
            )
            submitted = st.form_submit_button("Find Movies")
            if submitted:
                try:
                    st.write(genre_recommendations(title, num_recommendations))
                except KeyError:
                    st.warning("Please enter a valid title")

    # title based recommendations
    if option_2:
        cosine_sim, titles, indices = generate_tfidf()
        with st.form("title_selection"):
            st.markdown("### Title-based Recommendations")
            input_col1, input_col2 = st.columns(2)
            title = input_col1.text_input(
                "Movie Title",
                "Enter title...",
                help="Type in the movie you would like recommendations based upon",
            )
            num_recommendations = input_col2.number_input(
                "Number of Recommendations", min_value=1, max_value=50, value=10, step=1
            )
            submitted = st.form_submit_button("Find Movies")
            if submitted:
                try:
                    st.write(title_recommendations(title, num_recommendations))
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


if __name__ == "__main__":
    main()
