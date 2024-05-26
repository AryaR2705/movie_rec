import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

# Function to get BERT embeddings
def get_bert_embeddings(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1)  # Average pooling over tokens
    return embeddings

# Function to calculate cosine similarity
def calculate_similarity_scaled(embedding1, embedding2):
    dot_product = torch.sum(embedding1 * embedding2, dim=1)
    magnitude1 = torch.sqrt(torch.sum(embedding1 ** 2, dim=1))
    magnitude2 = torch.sqrt(torch.sum(embedding2 ** 2, dim=1))
    cosine_similarity = dot_product / (magnitude1 * magnitude2)
    
    # Scale cosine similarity to range between 0 and 5
    scaled_similarity = (cosine_similarity + 1) * 2.5
    return scaled_similarity.item()

# Load locally saved BERT model and tokenizer
model_path = "bert_base_uncased_model"
tokenizer_path = "bert_base_uncased_tokenizer"
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertModel.from_pretrained(model_path)

# Load que.csv and rec.csv
que_df = pd.read_csv("que.csv")
rec_df = pd.read_csv("rec.csv")

# Initialize list to store results
similarity_results = []

# Iterate through each row in rec.csv
for rec_index, rec_row in rec_df.iterrows():
    rec_title = rec_row['tags']
    rec_tags = rec_row['tags']
    
    # Get BERT embedding for rec_tags
    rec_embedding = get_bert_embeddings(rec_tags, model, tokenizer)
    
    # Iterate through each row in que.csv
    for que_index, que_row in que_df.iterrows():
        que_title = que_row['tags']
        que_tags = que_row['tags']
        
        # Get BERT embedding for que_tags
        que_embedding = get_bert_embeddings(que_tags, model, tokenizer)
        
        # Calculate similarity between rec_tags and que_tags
        similarity_score = calculate_similarity_scaled(rec_embedding, que_embedding)
        
        # Append results to the list
        similarity_results.append({'rec_title': rec_title, 'que_title': que_title, 'similarity_score': similarity_score})

# Convert results to DataFrame and sort by similarity_score in descending order
similarity_df = pd.DataFrame(similarity_results)
similarity_df = similarity_df.sort_values(by='similarity_score', ascending=False)

# Save results to CSV
similarity_df.to_csv('similarity_results.csv', index=False)


