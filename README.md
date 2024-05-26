# Movie_rec


A sophisticated movie recomendation system

DATA over 5000+ movies including tags, genre, synopsis, crew and more

Movie is recommended by 2 steps :
1) top 50 most similar films are choosen out of 5000 films
2) these 50 films are further filtered out by BERT model and finally TOP 5 movies are choosen

BERT is a Bidirectional Encoder Representations from Transformers, a generative model with more contextual understanding
it leverages it's power into movie recommendation system by further understanding contextual things like synopsis to provide more accurate result
