import numpy as np 
import pandas as pd
import ast
import pickle
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load data
movies = pd.read_csv('data/tmdb_5000_movies.csv')
credits = pd.read_csv('data/tmdb_5000_credits.csv')

# Merge datasets
movies = movies.merge(credits, on='title')

# Keeping important columns for recommendation
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 
                'vote_average', 'vote_count', 'popularity', 'release_date']]

# Clean data
movies.dropna(inplace=True)

# Convert string representation of lists to actual Python lists
def convert(text):
    if isinstance(text, str):
        try:
            return [i['name'] for i in ast.literal_eval(text)]
        except:
            return []
    return []

# Process genres
movies['genres'] = movies['genres'].apply(convert)

# Extract genre IDs for filtering
def extract_genre_ids(text):
    if isinstance(text, str):
        try:
            return [i['id'] for i in ast.literal_eval(text)]
        except:
            return []
    return []

# Modify to handle both string and list formats
movies['genre_ids'] = movies['genres'].apply(lambda x: extract_genre_ids(text=x) if isinstance(x, str) else [])

# Process keywords
movies['keywords'] = movies['keywords'].apply(convert)

# Process cast - keep top 3 actors
def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter += 1
    return L

movies['cast'] = movies['cast'].apply(convert_cast)

# Extract director name
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# Convert overview to list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])

# Remove spaces from strings in lists
def remove_space(L):
    return [i.replace(" ", "") for i in L]

movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)

# Create tags by concatenating features
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create a dataframe with selected columns
new_df = movies[['movie_id', 'title', 'tags', 'vote_average', 'vote_count', 
                'popularity', 'release_date', 'genres', 'overview']]

# Convert tags from list to string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower() if isinstance(x, str) else "")

# Apply stemming to tags
ps = PorterStemmer()
def stems(text):
    if not isinstance(text, str):
        return ""
    return " ".join([ps.stem(i) for i in text.split()])

new_df['tags'] = new_df['tags'].apply(stems)

# Extract year from release_date for filtering
new_df['year'] = pd.to_datetime(new_df['release_date'], errors='coerce').dt.year

# Convert original overview back to string for sentiment analysis
new_df['overview_text'] = movies['overview'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")

# Add mood classification based on overview sentiment
sid = SentimentIntensityAnalyzer()

def classify_mood(text):
    if not isinstance(text, str) or text == "":
        return "neutral"
    
    sentiment = sid.polarity_scores(text)
    compound = sentiment['compound']
    
    if compound >= 0.5:
        return "uplifting"
    elif compound >= 0.05:
        return "positive"
    elif compound <= -0.5:
        return "dark"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"
    
new_df['mood'] = new_df['overview_text'].apply(classify_mood)

# Create additional mood features using keywords in overview and genres
def mood_keywords(overview, genres):
    text = overview.lower() if isinstance(overview, str) else ""
    genre_text = " ".join(genres) if isinstance(genres, list) else ""
    
    # Define mood keyword associations
    action_words = ['action', 'thrill', 'adventure', 'exciting', 'fast', 'fight', 'battle', 'explosion']
    comedy_words = ['comedy', 'funny', 'laugh', 'hilarious', 'humor']
    romance_words = ['romance', 'love', 'relationship', 'passion', 'romantic']
    drama_words = ['drama', 'emotional', 'powerful', 'intense', 'moving']
    horror_words = ['horror', 'scary', 'terrifying', 'fear', 'creepy', 'nightmare']
    calm_words = ['calm', 'peaceful', 'gentle', 'quiet', 'relax']
    fantasy_words = ['fantasy', 'magic', 'wonder', 'mystical', 'supernatural']
    
    moods = []
    
    if any(word in text or word in genre_text for word in action_words):
        moods.append('action')
    if any(word in text or word in genre_text for word in comedy_words):
        moods.append('comedy')
    if any(word in text or word in genre_text for word in romance_words):
        moods.append('romance')
    if any(word in text or word in genre_text for word in drama_words):
        moods.append('drama')
    if any(word in text or word in genre_text for word in horror_words):
        moods.append('horror')
    if any(word in text or word in genre_text for word in calm_words):
        moods.append('calm')
    if any(word in text or word in genre_text for word in fantasy_words):
        moods.append('fantasy')
    
    if not moods:
        moods = ['general']
    
    return moods

new_df['mood_categories'] = new_df.apply(lambda x: mood_keywords(x['overview_text'], x['genres']), axis=1)

# Vectorize the tags
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()

# Calculate similarity
similarity = cosine_similarity(vector)

# Recommendation function with filtering and mood matching
def recommend(movie, new_df, similarity, top_n=5, min_rating=0.0, genres=None, year_range=None, mood=None):
    if movie not in new_df['title'].values:
        return [], [], []
    
    # Get the index of the movie
    index = new_df[new_df['title'] == movie].index[0]
    
    # Calculate similarity scores and sort
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    
    # Filter based on criteria
    filtered_movies = []
    for i, score in distances[1:]:  # Skip the first one as it's the movie itself
        movie_entry = new_df.iloc[i]
        
        # Rating filter
        if movie_entry['vote_average'] < min_rating:
            continue
        
        # Genre filter
        if genres and len(genres) > 0:
            if not any(genre.lower() in [g.lower() for g in movie_entry['genres']] for genre in genres):
                continue
                
        # Year range filter
        if year_range and len(year_range) == 2:
            if movie_entry['year'] < year_range[0] or movie_entry['year'] > year_range[1]:
                continue
                
        # Mood filter
        if mood:
            if mood.lower() not in [m.lower() for m in movie_entry['mood_categories']] and mood.lower() != movie_entry['mood'].lower():
                continue
                
        # Add to filtered movies
        filtered_movies.append((i, score))
        
        # Break if we have enough recommendations
        if len(filtered_movies) >= top_n:
            break
    
    # Extract information for recommended movies
    recommended_movie_names = []
    recommended_movie_ids = []
    similarity_scores = []
    
    for i, score in filtered_movies:
        recommended_movie_names.append(new_df.iloc[i]['title'])
        recommended_movie_ids.append(new_df.iloc[i]['movie_id'])
        similarity_scores.append(score)
    
    return recommended_movie_names, recommended_movie_ids, similarity_scores

# Create a lookup dictionary for all genre IDs and names
genres_list = []
for genres in movies['genres']:
    genres_list.extend(genres)
unique_genres = sorted(list(set(genres_list)))

# Create a dictionary of common moods
moods_dict = {
    'uplifting': 'Movies that leave you feeling inspired and optimistic',
    'positive': 'Light and cheerful movies with happy themes',
    'neutral': 'Balanced movies with a mix of emotions',
    'negative': 'More serious or somber movies',
    'dark': 'Intense, gritty or disturbing movies',
    'action': 'Exciting movies with thrills and adventure',
    'comedy': 'Funny and humorous movies',
    'romance': 'Movies about love and relationships',
    'drama': 'Emotional and character-driven stories',
    'horror': 'Scary and frightening movies',
    'calm': 'Peaceful and relaxing movies',
    'fantasy': 'Magical and imaginative stories'
}

# Save all necessary data
pickle.dump(new_df, open('artifacts/movie_list.pkl', 'wb'))
pickle.dump(similarity, open('artifacts/similarity.pkl', 'wb'))
pickle.dump(unique_genres, open('artifacts/genres.pkl', 'wb'))
pickle.dump(moods_dict, open('artifacts/moods.pkl', 'wb'))

# Save the recommend function separately for direct import
import dill
dill.dump(lambda movie, top_n=5, min_rating=0.0, genres=None, year_range=None, mood=None: recommend(movie, new_df, similarity, top_n, min_rating, genres, year_range, mood), open('artifacts/recommend_function.pkl', 'wb'))

print("Training completed and artifacts saved successfully!")