import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pickle
import requests
import streamlit as st
from PIL import Image
from io import BytesIO

# Global variables
TMDB_API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual TMDB API key
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Define mood keywords mapping
MOOD_KEYWORDS = {
    'Happy': ['happy', 'comedy', 'joy', 'funny', 'hilarious', 'uplifting', 'cheerful', 'light-hearted', 'feel-good', 'romance'],
    'Sad': ['sad', 'tragedy', 'emotional', 'drama', 'tear', 'heartbreak', 'melancholy', 'grief', 'sorrow', 'depressing'],
    'Thrilling': ['thrill', 'suspense', 'tension', 'action', 'adventure', 'crime', 'mystery', 'exciting', 'fast-paced', 'adrenaline'],
    'Scary': ['horror', 'scary', 'terrifying', 'fear', 'frightening', 'creepy', 'eerie', 'supernatural', 'monster', 'nightmare'],
    'Thought-provoking': ['philosophical', 'deep', 'thought-provoking', 'psychological', 'complex', 'intellectual', 'meaning', 'profound', 'existential', 'mind-bending'],
    'Inspiring': ['inspire', 'motivation', 'overcome', 'achievement', 'success', 'biographical', 'true story', 'courage', 'triumph', 'underdog']
}

class MovieRecommender:
    def __init__(self):
        self.movies_df = None
        self.similarity_matrix = None
        
    def load_and_preprocess_data(self, movies_path, credits_path=None):
        """Load and preprocess TMDB dataset."""
        try:
            # Load the movies dataset
            movies_df = pd.read_csv(movies_path)
            
            # Check if 'id' column exists, if not, create one
            if 'id' not in movies_df.columns:
                print("Warning: 'id' column not found. Creating an index-based ID column.")
                movies_df['id'] = range(1, len(movies_df) + 1)
                
            # Load credits if provided and if it exists
            if credits_path:
                try:
                    credits_df = pd.read_csv(credits_path)
                    # Check if both dataframes have an 'id' column for merging
                    if 'id' in credits_df.columns:
                        movies_df = movies_df.merge(credits_df, on='id', how='left')
                    else:
                        print("Warning: 'id' column not found in credits file. Skipping merge.")
                except Exception as e:
                    print(f"Error loading credits file: {e}. Proceeding without it.")
            
            # Ensure essential columns exist
            essential_columns = ['title', 'overview']
            for col in essential_columns:
                if col not in movies_df.columns:
                    raise ValueError(f"Essential column '{col}' not found in dataset.")
                    
            # Keep only necessary columns that exist in the dataframe
            desired_columns = ['id', 'title', 'overview', 'genres', 'keywords', 'release_date', 
                              'vote_average', 'vote_count', 'runtime', 'original_language']
            
            columns_to_keep = [col for col in desired_columns if col in movies_df.columns]
            self.movies_df = movies_df[columns_to_keep].copy()
            
            # Print some info about the loaded data
            print(f"Loaded dataset with {len(self.movies_df)} movies and {len(columns_to_keep)} columns.")
            print(f"Columns: {columns_to_keep}")
            
            # Clean and transform data
            self._clean_data()
            self._extract_year()
            self._process_genres()
            self._assign_moods()
            self._create_combined_features()
            
            return self.movies_df
            
        except Exception as e:
            print(f"Error in load_and_preprocess_data: {e}")
            raise
        
    def _clean_data(self):
        """Clean the dataframe by handling missing values."""
        try:
            # Fill missing values
            if 'overview' in self.movies_df.columns:
                self.movies_df['overview'] = self.movies_df['overview'].fillna('')
            else:
                self.movies_df['overview'] = ''
                
            if 'genres' in self.movies_df.columns:
                self.movies_df['genres'] = self.movies_df['genres'].fillna('[]')
                
            if 'keywords' in self.movies_df.columns:
                self.movies_df['keywords'] = self.movies_df['keywords'].fillna('[]')
                
            # Remove rows with empty titles
            self.movies_df = self.movies_df[self.movies_df['title'].notna()]
        except Exception as e:
            print(f"Error in _clean_data: {e}")
    
    def _extract_year(self):
        """Extract year from release_date."""
        try:
            if 'release_date' in self.movies_df.columns:
                self.movies_df['year'] = pd.to_datetime(self.movies_df['release_date'], 
                                                       errors='coerce').dt.year
            else:
                self.movies_df['year'] = np.nan
        except Exception as e:
            print(f"Error in _extract_year: {e}")
            self.movies_df['year'] = np.nan
    
    def _extract_value_from_json(self, json_str, key='name'):
        """Extract values from JSON-like strings."""
        try:
            import ast
            # Check if the string is already properly formatted
            if isinstance(json_str, str):
                # Handle potential JSON formatting issues
                try:
                    result = ast.literal_eval(json_str)
                    if isinstance(result, list):
                        return [item[key] for item in result if key in item]
                    return []
                except (SyntaxError, ValueError):
                    # If literal_eval fails, try a simple string-based extraction for formats like "[{'name': 'Action'}]"
                    matches = re.findall(r"'name':\s*'([^']+)'", json_str)
                    return matches if matches else []
            return []
        except Exception as e:
            print(f"Error extracting values: {e}")
            return []
    
    def _process_genres(self):
        """Process genres from JSON format to string list."""
        try:
            if 'genres' in self.movies_df.columns:
                # Convert genres from JSON to list of genre names
                self.movies_df['genres_list'] = self.movies_df['genres'].apply(
                    lambda x: self._extract_value_from_json(x) if isinstance(x, str) else [])
                
                # Convert list to space-separated string for TF-IDF
                self.movies_df['genres_str'] = self.movies_df['genres_list'].apply(
                    lambda x: ' '.join([genre.lower().replace(' ', '_') for genre in x]))
            else:
                self.movies_df['genres_list'] = [[]] * len(self.movies_df)
                self.movies_df['genres_str'] = ''
            
            if 'keywords' in self.movies_df.columns:
                # Process keywords similarly
                self.movies_df['keywords_list'] = self.movies_df['keywords'].apply(
                    lambda x: self._extract_value_from_json(x) if isinstance(x, str) else [])
                self.movies_df['keywords_str'] = self.movies_df['keywords_list'].apply(
                    lambda x: ' '.join([kw.lower().replace(' ', '_') for kw in x]))
            else:
                self.movies_df['keywords_list'] = [[]] * len(self.movies_df)
                self.movies_df['keywords_str'] = ''
        except Exception as e:
            print(f"Error in _process_genres: {e}")
            self.movies_df['genres_list'] = [[]] * len(self.movies_df)
            self.movies_df['genres_str'] = ''
            self.movies_df['keywords_list'] = [[]] * len(self.movies_df)
            self.movies_df['keywords_str'] = ''
    
    def _assign_moods(self):
        """Assign mood tags based on overview, genres, and keywords."""
        try:
            # Initialize mood columns
            for mood in MOOD_KEYWORDS:
                self.movies_df[f'mood_{mood.lower()}'] = 0
            
            # Function to check for mood keywords in text
            def check_mood(text, mood_keywords):
                if not isinstance(text, str):
                    return 0
                text = text.lower()
                return sum(1 for kw in mood_keywords if kw.lower() in text)
            
            # Assign mood scores based on overview, genres, and keywords
            for mood, keywords in MOOD_KEYWORDS.items():
                # Check in overview
                if 'overview' in self.movies_df.columns:
                    self.movies_df[f'mood_{mood.lower()}'] += self.movies_df['overview'].apply(
                        lambda x: check_mood(x, keywords))
                
                # Check in genres
                if 'genres_str' in self.movies_df.columns:
                    self.movies_df[f'mood_{mood.lower()}'] += self.movies_df['genres_str'].apply(
                        lambda x: check_mood(x, keywords))
                
                # Check in keywords if available
                if 'keywords_str' in self.movies_df.columns:
                    self.movies_df[f'mood_{mood.lower()}'] += self.movies_df['keywords_str'].apply(
                        lambda x: check_mood(x, keywords))
            
            # Create a combined mood string for each movie
            self.movies_df['moods'] = ''
            for mood in MOOD_KEYWORDS:
                # Add mood to the movie's mood string if its score is above threshold
                threshold = 1  # Adjust as needed
                mask = self.movies_df[f'mood_{mood.lower()}'] >= threshold
                self.movies_df.loc[mask, 'moods'] += f"{mood.lower()} "
            
            self.movies_df['moods'] = self.movies_df['moods'].str.strip()
        except Exception as e:
            print(f"Error in _assign_moods: {e}")
            self.movies_df['moods'] = ''
    
    def _create_combined_features(self):
        """Create a combined feature string for similarity calculation."""
        try:
            # Combine overview, genres, and moods for similarity calculation
            self.movies_df['combined_features'] = ''
            
            if 'overview' in self.movies_df.columns:
                self.movies_df['combined_features'] += self.movies_df['overview'].fillna('') + ' '
            
            if 'genres_str' in self.movies_df.columns:
                # Add genres with higher weight (repeat them)
                self.movies_df['combined_features'] += self.movies_df['genres_str'].fillna('') + ' ' + self.movies_df['genres_str'].fillna('') + ' '
            
            if 'keywords_str' in self.movies_df.columns:
                self.movies_df['combined_features'] += self.movies_df['keywords_str'].fillna('') + ' '
            
            # Add moods to combined features
            self.movies_df['combined_features'] += self.movies_df['moods'].fillna('')
            
            # Clean the combined features
            self.movies_df['combined_features'] = self.movies_df['combined_features'].apply(
                lambda x: x.lower().strip() if isinstance(x, str) else '')
        except Exception as e:
            print(f"Error in _create_combined_features: {e}")
            self.movies_df['combined_features'] = self.movies_df['overview'].fillna('')
    
    def compute_similarity_matrix(self):
        """Compute the similarity matrix based on combined features."""
        try:
            # Create TF-IDF vectors
            tfidf = TfidfVectorizer(stop_words='english')
            
            # Replace NaN values with empty string
            features = self.movies_df['combined_features'].fillna('')
            
            # Create TF-IDF matrix
            tfidf_matrix = tfidf.fit_transform(features)
            
            # Compute cosine similarity
            self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            return self.similarity_matrix
        except Exception as e:
            print(f"Error in compute_similarity_matrix: {e}")
            raise
    
    def get_recommendations(self, movie_title, n=10, mood=None, min_year=None, max_year=None, language=None, genres=None):
        """Get movie recommendations based on similarity and filters."""
        try:
            # Check if movie exists in our database
            movie_index = self.movies_df[self.movies_df['title'] == movie_title].index
            
            if len(movie_index) == 0:
                return pd.DataFrame()  # Return empty dataframe if movie not found
            
            movie_index = movie_index[0]
            
            # Get similarity scores for the movie
            similarity_scores = list(enumerate(self.similarity_matrix[movie_index]))
            
            # Sort based on similarity score
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            
            # Get top n+1 similar movies (including the movie itself)
            top_movies_indices = [i[0] for i in similarity_scores[:100]]  # Get 100 to allow for filtering
            
            # Create a dataframe of recommended movies
            recommended_df = self.movies_df.iloc[top_movies_indices].copy()
            
            # Add similarity score
            recommended_df['similarity'] = [i[1] for i in similarity_scores[:100]]
            
            # Filter by mood if specified
            if mood and mood.lower() != 'any':
                mood_col = f'mood_{mood.lower()}'
                if mood_col in recommended_df.columns:
                    recommended_df = recommended_df[recommended_df[mood_col] > 0]
            
            # Apply year filter if specified
            if min_year and 'year' in recommended_df.columns:
                recommended_df = recommended_df[recommended_df['year'] >= min_year]
            if max_year and 'year' in recommended_df.columns:
                recommended_df = recommended_df[recommended_df['year'] <= max_year]
            
            # Apply language filter if specified
            if language and language != 'Any' and 'original_language' in recommended_df.columns:
                recommended_df = recommended_df[recommended_df['original_language'] == language.lower()]
            
            # Apply genre filter if specified
            if genres and genres != 'Any' and 'genres_str' in recommended_df.columns:
                # Filter movies that have the specified genre
                recommended_df = recommended_df[recommended_df['genres_str'].str.contains(genres.lower().replace(' ', '_'), na=False)]
            
            # Remove the input movie itself
            recommended_df = recommended_df[recommended_df.index != movie_index]
            
            # Sort by similarity and then by rating
            if 'vote_average' in recommended_df.columns:
                # Normalize rating influence
                recommended_df['score'] = (0.8 * recommended_df['similarity']) + (0.2 * (recommended_df['vote_average'] / 10))
                recommended_df = recommended_df.sort_values('score', ascending=False)
            else:
                recommended_df = recommended_df.sort_values('similarity', ascending=False)
            
            # Return top n recommendations
            return recommended_df.head(n)
        except Exception as e:
            print(f"Error in get_recommendations: {e}")
            return pd.DataFrame()
    
    def get_movie_poster(self, movie_id):
        """Get movie poster from TMDB API."""
        try:
            response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US")
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return f"{POSTER_BASE_URL}{poster_path}"
            return None
        except Exception as e:
            print(f"Error getting movie poster: {e}")
            return None
    
    def save_model(self, filename='movie_recommender.pkl'):
        """Save the model to a pickle file."""
        try:
            model_data = {
                'movies_df': self.movies_df,
                'similarity_matrix': self.similarity_matrix
            }
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {filename}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    @classmethod
    def load_model(cls, filename='movie_recommender.pkl'):
        """Load the model from a pickle file."""
        recommender = cls()
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
                recommender.movies_df = model_data['movies_df']
                recommender.similarity_matrix = model_data['similarity_matrix']
            print(f"Model loaded from {filename}")
            return recommender
        except Exception as e:
            print(f"Error loading model from {filename}: {e}")
            return None


# Streamlit UI
def create_ui():
    st.set_page_config(page_title="Movie Recommender", layout="wide")
    
    st.title("Movie Recommendation System")
    st.subheader("Find movies similar to your favorites!")
    
    # Allow user to upload dataset
    with st.sidebar:
        st.header("Dataset Options")
        uploaded_movies = st.file_uploader("Upload movies dataset (CSV)", type="csv")
        uploaded_credits = st.file_uploader("Upload credits dataset (CSV)", type="csv", help="Optional")
        
        use_uploaded = False
        if uploaded_movies is not None:
            use_uploaded = st.checkbox("Use uploaded dataset", value=True)
        
        if st.button("Reset Recommender"):
            if 'recommender' in st.session_state:
                del st.session_state['recommender']
            if 'recommendations' in st.session_state:
                del st.session_state['recommendations']
            st.success("Recommender reset! Please refresh the page.")
    
    # Check if recommender is already loaded in session state
    if 'recommender' not in st.session_state:
        with st.spinner("Loading movie data... This might take a moment."):
            try:
                # Try to load saved model first if not using uploaded files
                recommender = None
                if not use_uploaded:
                    recommender = MovieRecommender.load_model()
                
                if recommender is None:
                    # If loading fails or using uploaded files, create from scratch
                    recommender = MovieRecommender()
                    
                    if use_uploaded and uploaded_movies is not None:
                        # Use uploaded files
                        movies_path = uploaded_movies
                        credits_path = uploaded_credits if uploaded_credits is not None else None
                    else:
                        # Use default files
                        try:
                            # Try with TMDB 5000 dataset
                            movies_path = 'tmdb_5000_movies.csv'
                            credits_path = 'tmdb_5000_credits.csv'
                            
                            # Check if files exist, if not try alternative filenames
                            import os
                            if not os.path.exists(movies_path):
                                alternative_paths = ['movies.csv', 'tmdb_movies.csv', 'movie_data.csv']
                                for path in alternative_paths:
                                    if os.path.exists(path):
                                        movies_path = path
                                        break
                            
                            if credits_path and not os.path.exists(credits_path):
                                credits_path = None  # Skip credits if file doesn't exist
                        except Exception as e:
                            st.error(f"Error finding dataset files: {e}")
                            return
                    
                    recommender.load_and_preprocess_data(movies_path, credits_path)
                    recommender.compute_similarity_matrix()
                    
                    # Save model for future use if not using uploaded files
                    if not use_uploaded:
                        recommender.save_model()
                
                st.session_state['recommender'] = recommender
            except Exception as e:
                st.error(f"Error loading data: {e}")
                st.info("Please upload your own movie dataset or check if the default dataset files are available.")
                return
    
    recommender = st.session_state['recommender']
    
    # Get unique languages
    if 'original_language' in recommender.movies_df.columns:
        languages = ['Any'] + sorted(recommender.movies_df['original_language'].dropna().unique().tolist())
    else:
        languages = ['Any']
    
    # Get unique genres from genres_list
    all_genres = set()
    for genres in recommender.movies_df['genres_list']:
        if isinstance(genres, list):
            all_genres.update(genres)
    genres_list = ['Any'] + sorted(list(all_genres))
    
    # Set up columns for filters
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Filters")
        
        # Movie selection
        movie_list = recommender.movies_df['title'].tolist()
        selected_movie = st.selectbox("Select a movie:", movie_list)
        
        # Mood filter
        moods = ['Any'] + list(MOOD_KEYWORDS.keys())
        selected_mood = st.selectbox("Movie mood:", moods)
        
        # Year range
        if 'year' in recommender.movies_df.columns:
            min_year = int(recommender.movies_df['year'].dropna().min())
            max_year = int(recommender.movies_df['year'].dropna().max())
        else:
            min_year, max_year = 1900, 2023
            
        year_range = st.slider("Year range:", min_year, max_year, (min_year, max_year))
        
        # Language filter
        selected_language = st.selectbox("Language:", languages)
        
        # Genre filter
        selected_genre = st.selectbox("Genre:", genres_list)
        
        # Number of recommendations
        num_recommendations = st.slider("Number of recommendations:", 1, 20, 5)
        
        # Get recommendations button
        if st.button("Find Recommendations"):
            if selected_movie:
                with st.spinner("Finding similar movies..."):
                    recommendations = recommender.get_recommendations(
                        selected_movie, 
                        n=num_recommendations,
                        mood=selected_mood if selected_mood != 'Any' else None,
                        min_year=year_range[0],
                        max_year=year_range[1],
                        language=selected_language if selected_language != 'Any' else None,
                        genres=selected_genre if selected_genre != 'Any' else None
                    )
                    
                    if recommendations.empty:
                        st.warning("No recommendations found with these filters. Try changing your criteria.")
                    else:
                        st.session_state['recommendations'] = recommendations
            else:
                st.warning("Please select a movie first.")
    
    # Display recommendations
    with col2:
        st.subheader("Recommendations")
        
        if 'recommendations' in st.session_state:
            recommendations = st.session_state['recommendations']
            
            # Create a grid layout
            num_cols = 3
            recommendations_list = recommendations.to_dict('records')
            
            for i in range(0, len(recommendations_list), num_cols):
                cols = st.columns(num_cols)
                for j in range(num_cols):
                    idx = i + j
                    if idx < len(recommendations_list):
                        movie = recommendations_list[idx]
                        with cols[j]:
                            st.subheader(movie['title'])
                            
                            # Display poster if available and API key is provided
                            if TMDB_API_KEY != "8265bd1679663a7ea12ac168da84d2e8" and 'id' in movie:
                                poster_url = recommender.get_movie_poster(movie['id'])
                                if poster_url:
                                    try:
                                        response = requests.get(poster_url)
                                        img = Image.open(BytesIO(response.content))
                                        st.image(img, width=200)
                                    except Exception as e:
                                        st.image("https://via.placeholder.com/200x300?text=No+Poster", width=200)
                                else:
                                    st.image("https://via.placeholder.com/200x300?text=No+Poster", width=200)
                            else:
                                st.image("https://via.placeholder.com/200x300?text=No+Poster", width=200)
                            
                            # Display movie info
                            if 'year' in movie and not pd.isna(movie['year']):
                                st.write(f"**Year:** {int(movie['year'])}")
                            
                            if 'vote_average' in movie:
                                st.write(f"**Rating:** {movie['vote_average']:.1f}/10")
                            
                            if 'genres_list' in movie and movie['genres_list']:
                                st.write(f"**Genres:** {', '.join(movie['genres_list'])}")
                            
                            if 'moods' in movie and movie['moods']:
                                st.write(f"**Mood:** {movie['moods'].title()}")
                            
                            # Display overview with a "Read more" button
                            if 'overview' in movie and movie['overview']:
                                with st.expander("Overview"):
                                    st.write(movie['overview'])


# If running this script directly
if __name__ == "__main__":
    create_ui()