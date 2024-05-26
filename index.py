# Import necessary libraries
from flask import Flask, request, render_template
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import pickle

# Initialize Flask app
app = Flask(__name__)

# Function to get BERT embeddings
def get_bert_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)  # Average pooling over tokens
    return embeddings

# Function to calculate cosine similarity
def calculate_similarity(embedding1, embedding2):
    dot_product = torch.sum(embedding1 * embedding2, dim=1)
    magnitude1 = torch.sqrt(torch.sum(embedding1 ** 2, dim=1))
    magnitude2 = torch.sqrt(torch.sum(embedding2 ** 2, dim=1))
    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    return cosine_similarity.item()

# Load locally saved BERT model and tokenizer
model_path = "bert_base_uncased_model"
tokenizer_path = "bert_base_uncased_tokenizer"
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertModel.from_pretrained(model_path)

# Load movie dictionary and similarity scores
movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl', 'rb'))

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

def recommend(movie):
    selected_movie_tag = movies[movies['title'] == movie]['tags'].values[0]
    recommendations_data = {'title': [], 'tags': []}  # Data to store recommendations
    
    # Write selected movie and its tags to que.csv
    with open('que.csv', 'w') as file:
        file.write("title,tags\n")
        file.write(f'"{movie}","{selected_movie_tag}"\n')
    
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:15]

    recommended_movies = []

    # Write recommended movies and their tags to rec.csv
    with open('rec.csv', 'w') as file:
        file.write("title,tags\n")
        for i, movie_info in enumerate(movies_list):
            movie_title = movies.iloc[movie_info[0]]['title']  # Get the title of the recommended movie
            recommended_movies.append(movie_title)
            tags = movies.iloc[movie_info[0]]['tags']  # Get tags of the recommended movie
            recommendations_data['title'].append(movie_title)
            recommendations_data['tags'].append(tags)
            file.write(f'"{movie_title}","{tags}"\n')
    
    return recommended_movies

@app.route('/recommend', methods=['POST'])
def recommended_movies():
    selected_movie_name = request.form['movie_name']
    recommended_movie_titles = recommend(selected_movie_name)
    
    # Run similarity calculations
    run_similarity_calculations()
    
    # Read similarity results from CSV
    similarity_results = pd.read_csv('similarity_results.csv')
    
    return render_template('recommendations.html', recommended_movies=recommended_movie_titles, similarity_results=similarity_results)
# Function to run similarity calculations
def run_similarity_calculations():
    # Load que.csv and rec.csv
    que_df = pd.read_csv("que.csv")
    rec_df = pd.read_csv("rec.csv")

    # Initialize list to store results
    similarity_results = []

    # Iterate through each row in rec.csv
    for rec_index, rec_row in rec_df.iterrows():
        rec_title = rec_row['title']
        rec_tags = rec_row['tags']
        
        # Get BERT embedding for rec_tags
        rec_embedding = get_bert_embeddings(rec_tags, model, tokenizer)
        
        # Iterate through each row in que.csv
        for que_index, que_row in que_df.iterrows():
            que_title = que_row['title']
            que_tags = que_row['tags']
            
            # Get BERT embedding for que_tags
            que_embedding = get_bert_embeddings(que_tags, model, tokenizer)
            
            # Calculate similarity between rec_tags and que_tags
            similarity_score = calculate_similarity(rec_embedding, que_embedding)
            
            # Append results to the list
            similarity_results.append({'rec_title': rec_title, 'que_title': que_title, 'similarity_score': similarity_score})

    # Convert results to DataFrame and sort by similarity_score in descending order
    similarity_df = pd.DataFrame(similarity_results)
    similarity_df = similarity_df.sort_values(by='similarity_score', ascending=False)

    # Save results to CSV
    similarity_df.to_csv('similarity_results.csv', index=False)

if __name__ == '__main__':
    app.run(debug=True)

    # Run similarity calculations after starting Flask app
    run_similarity_calculations()
