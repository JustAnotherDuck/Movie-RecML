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
TMDB_API_KEY = "f74484e522f47f9813841442dc4ae45a"  # Your provided TMDB API key
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER_IMAGE = "https://via.placeholder.com/200x300?text=No+Poster"

# Define mood keywords mapping
MOOD_KEYWORDS = {
    'Happy': ['happy', 'comedy', 'joy', 'funny', 'hilarious', 'uplifting', 'cheerful', 'light-hearted', 'feel-good', 'romance'],
    'Sad': ['sad', 'tragedy', 'emotional', 'drama', 'tear', 'heartbreak', 'melancholy', 'grief', 'sorrow', 'depressing'],
    'Thrilling': ['thrill', 'suspense', 'tension', 'action', 'adventure', 'crime', 'mystery', 'exciting', 'fast-paced', 'adrenaline'],
    'Scary': ['horror', 'scary', 'terrifying', 'fear', 'frightening', 'creepy', 'eerie', 'supernatural', 'monster', 'nightmare'],
    'Thought-provoking': ['philosophical', 'deep', 'thought-provoking', 'psychological', 'complex', 'intellectual', 'meaning', 'profound', 'existential', 'mind-bending'],
    'Inspiring': ['inspire', 'motivation', 'overcome', 'achievement', 'success', 'biographical', 'true story', 'courage', 'triumph', 'underdog']
}

# Custom CSS for styling
def load_css():
    st.markdown("""
    <style>
    .movie-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s;
    }
    .movie-card:hover {
        transform: translateY(-5px);
    }
    .movie-poster {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .movie-title {
        color: #1a1a1a;
        font-size: 20px;
        font-weight: bold;
        margin-top: 10px;
        margin-bottom: 5px;
    }
    .movie-info {
        color: #444;
        font-size: 14px;
        margin-bottom: 3px;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin-right: 5px;
        margin-bottom: 5px;
        color: white;
    }
    .badge-genre {
        background-color: #3498db;
    }
    .badge-mood {
        background-color: #9b59b6;
    }
    .section-divider {
        height: 3px;
        background: linear-gradient(to right, #6a11cb, #2575fc);
        margin: 20px 0;
        border-radius: 3px;
    }
    .sidebar-header {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 15px;
        border-bottom: 2px solid #2575fc;
        padding-bottom: 10px;
    }
    .filter-section {
        background-color: #f1f3f6;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .recommendations-header {
        background-color: #f1f3f6;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        border-left: 5px solid #2575fc;
    }
    .overview-text {
        font-size: 14px;
        line-height: 1.5;
        color: #555;
    }
    </style>
    """, unsafe_allow_html=True)

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
            # Make API request to TMDB
            response = requests.get(
                f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US",
                timeout=5
            )
            # Check if request was successful
            if response.status_code == 200:
                data = response.json()
                poster_path = data.get('poster_path')
                if poster_path:
                    return f"{POSTER_BASE_URL}{poster_path}"
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching movie poster for movie ID {movie_id}: {e}")
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
    # Set page configuration
    st.set_page_config(
        page_title="Movie Recommender",
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_css()
    
    # App title and introduction with nicer styling
    st.markdown("<h1 style='text-align: center; color: #2575fc;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Find movies similar to your favorites or explore by mood!</p>", unsafe_allow_html=True)
    
    # Add a visual divider
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Allow user to upload dataset
    with st.sidebar:
        st.markdown("<div class='sidebar-header'>Dataset Options</div>", unsafe_allow_html=True)
        
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
        with st.spinner("🔄 Loading movie data... This might take a moment."):
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
    
    # Set up columns for filters and recommendations
    col1, col2 = st.columns([1, 3])
    
    # Filters column
    with col1:
        st.markdown("<div class='sidebar-header'>🔍 Filters</div>", unsafe_allow_html=True)
        
        with st.container():
            st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
            # Movie selection with search box
            movie_list = recommender.movies_df['title'].tolist()
            selected_movie = st.selectbox("Select a movie:", movie_list)
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='filter-section'>", unsafe_allow_html=True)
        # Mood filter with nicer labels
        moods = ['Any'] + list(MOOD_KEYWORDS.keys())
        selected_mood = st.selectbox("Movie mood:", moods)
        
        # Year range with more modern styling
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
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Get recommendations button with better styling
        if st.button("🔍 Find Recommendations", use_container_width=True):
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
    
    # Recommendations column
    with col2:
        st.markdown("<div class='recommendations-header'><h2>🎬 Recommendations</h2></div>", unsafe_allow_html=True)
        
        if 'recommendations' in st.session_state:
            recommendations = st.session_state['recommendations']
            
            # Create a grid layout with better styling
            num_cols = 3
            recommendations_list = recommendations.to_dict('records')
            
            # Display chosen movie
            if selected_movie:
                st.markdown(f"<p style='font-size: 16px; margin-bottom: 20px;'>Movies similar to <b>{selected_movie}</b></p>", unsafe_allow_html=True)
            
            # Add a visual divider
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Display movies in a grid
            for i in range(0, len(recommendations_list), num_cols):
                cols = st.columns(num_cols)
                for j in range(num_cols):
                    idx = i + j
                    if idx < len(recommendations_list):
                        movie = recommendations_list[idx]
                        with cols[j]:
                            # Start of movie card with CSS styling
                            st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
                            
                            # Title with proper styling
                            st.markdown(f"<div class='movie-title'>{movie['title']}</div>", unsafe_allow_html=True)
                            
                            # Display poster with rounded corners
                            if 'id' in movie:
                                poster_url = recommender.get_movie_poster(movie['id'])
                                if poster_url:
                                    try:
                                        response = requests.get(poster_url, timeout=5)
                                        response.raise_for_status()
                                        img = Image.open(BytesIO(response.content))
                                        st.markdown("<div class='movie-poster'>", unsafe_allow_html=True)
                                        st.image(img, width=200, use_container_width=True)
                                        st.markdown("</div>", unsafe_allow_html=True)
                                    except (requests.exceptions.RequestException, Exception) as e:
                                        print(f"Error loading poster image for {movie['title']}: {e}")
                                        st.markdown("<div class='movie-poster'>", unsafe_allow_html=True)
                                        st.image(PLACEHOLDER_IMAGE, use_container_width=True)
                                        st.markdown("</div>", unsafe_allow_html=True)
                                else:
                                    st.markdown("<div class='movie-poster'>", unsafe_allow_html=True)
                                    st.image(PLACEHOLDER_IMAGE, use_container_width=True)
                                    st.markdown("</div>", unsafe_allow_html=True)
                            else:
                                st.markdown("<div class='movie-poster'>", unsafe_allow_html=True)
                                st.image(PLACEHOLDER_IMAGE, use_container_width=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Movie info with consistent styling
                            info_html = ""
                            
                            # Year info
                            if 'year' in movie and not pd.isna(movie['year']):
                                info_html += f"<div class='movie-info'><b>Year:</b> {int(movie['year'])}</div>"
                            
                            # Rating info
                            if 'vote_average' in movie:
                                stars = "⭐" * round(movie['vote_average'] / 2)
                                info_html += f"<div class='movie-info'><b>Rating:</b> {movie['vote_average']:.1f}/10 {stars}</div>"
                            
                            # Genres info with badges
                            if 'genres_list' in movie and movie['genres_list']:
                                info_html += f"<div class='movie-info'><b>Genres:</b> "
                                for genre in movie['genres_list'][:3]:  # Limit to 3 genres to prevent overflow
                                    info_html += f"<span class='badge badge-genre'>{genre}</span>"
                                info_html += "</div>"
                            
                            # Mood info with badges
                            if 'moods' in movie and movie['moods']:
                                info_html += f"<div class='movie-info'><b>Mood:</b> "
                                for mood in movie['moods'].split()[:2]:  # Limit to 2 moods
                                    info_html += f"<span class='badge badge-mood'>{mood.title()}</span>"
                                info_html += "</div>"
                            
                            # Display all info
                            st.markdown(info_html, unsafe_allow_html=True)
                            
                            # Display overview with a "Read more" button and better styling
                            if 'overview' in movie and movie['overview']:
                                with st.expander("Overview"):
                                    st.markdown(f"<div class='overview-text'>{movie['overview']}</div>", unsafe_allow_html=True)
                            
                            # End of movie card
                            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Display placeholder when no recommendations are available
            st.markdown("""
                <div style="text-align: center; padding: 50px; color: #777;">
                    <h3>Select a movie and click "Find Recommendations" to see similar movies</h3>
                    <p>Use the filters on the left to refine your search</p>
                </div>
            """, unsafe_allow_html=True)


# If running this script directly
if __name__ == "__main__":
    create_ui()