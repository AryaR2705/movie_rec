import pandas as pd
import streamlit as st
import pickle

def recommend(movie):
    selected_movie_tag = movies[movies['title'] == movie]['tags'].values[0]
    recommendations_data = {'title': [], 'tags': []}  # Data to store recommendations
    with open('que.csv', 'w') as file:
        file.write("title,tags\n")
        file.write(f'"{movie}","{selected_movie_tag}"\n')  # Write selected movie and its tags

    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:15]

    recommended_movies = []

    with open('rec.csv', 'w') as file:
        file.write("title,tags\n")
        for i, movie_info in enumerate(movies_list):
            movie_title = movies.iloc[movie_info[0]]['title']  # Get the title of the recommended movie
            recommended_movies.append(movie_title)
            tags = movies.iloc[movie_info[0]]['tags']  # Get tags of the recommended movie
            recommendations_data['title'].append(movie_title)
            recommendations_data['tags'].append(tags)
            file.write(f'"{movie_title}","{tags}"\n')  # Write recommended movie and its tags
    
    return recommended_movies

movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl', 'rb'))

st.title('Movie Recommender System')

selected_movie_name = st.selectbox(
    'Search For a Movie Name',
    movies['title'].values
)

if st.button('Recommend'):
    recommended_movie_titles = recommend(selected_movie_name)
    st.write("Recommended Movies:")
    for movie_title in recommended_movie_titles:
        st.write(movie_title)
