'''import httpx
import streamlit as st
import pickle
import pandas as pd
import dill
from datetime import datetime
import random

# Set page config for better appearance
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (unchanged)
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .movie-title {
        font-weight: bold;
        text-align: center;
        height: 3em;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .recommendation-header {
        font-size: 1.8rem;
        color: #0083B8;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        width: 100%;
    }
    .mood-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        cursor: pointer;
        transition: transform 0.3s;
        height: 100%;
    }
    .mood-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .mood-emoji {
        font-size: 2rem;
        margin-bottom: 10px;
    }
    .stSelectbox>div>div {
        background-color: #F0F0F0;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to fetch movie poster
@st.cache_data
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        with httpx.Client() as client:
            response = client.get(url)
        if response.status_code != 200:
            return "https://via.placeholder.com/500x750?text=No+Poster+Available"
        
        data = response.json()
        if 'poster_path' not in data or data['poster_path'] is None:
            return "https://via.placeholder.com/500x750?text=No+Poster+Available"
            
        poster_path = data['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        return full_path
    except Exception as e:
        st.error(f"Error fetching poster for movie_id {movie_id}: {e}")
        return "https://via.placeholder.com/500x750?text=No+Poster+Available"

# Function to fetch movie details
@st.cache_data
def fetch_movie_details(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        with httpx.Client() as client:
            response = client.get(url)
        if response.status_code != 200:
            return {}
        
        data = response.json()
        return {
            'title': data.get('title', 'Unknown'),
            'overview': data.get('overview', 'No description available.'),
            'release_date': data.get('release_date', 'Unknown'),
            'rating': data.get('vote_average', 0),
            'genres': ', '.join([g['name'] for g in data.get('genres', [])])
        }
    except Exception as e:
        st.error(f"Error fetching details for movie_id {movie_id}: {e}")
        return {}

# Load data
@st.cache_resource
def load_data():
    try:
        movies = pickle.load(open('artifacts/movie_list.pkl', 'rb'))
        similarity = pickle.load(open('artifacts/similarity.pkl', 'rb'))
        genres = pickle.load(open('artifacts/genres.pkl', 'rb'))
        moods = pickle.load(open('artifacts/moods.pkl', 'rb'))
        recommend_func = dill.load(open('artifacts/recommend_function.pkl', 'rb'))
        return movies, similarity, genres, moods, recommend_func
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

# Modified recommendation function
def recommend(movie, min_rating=0.0, genres=None, year_range=None, mood=None):
    try:
        movies, similarity, all_genres, moods_dict, recommend_func = load_data()
        if movies is None:
            st.error("Failed to load movie data")
            return [], [], []
        
        # Use the imported recommend function
        recommended_movie_names, recommended_movie_ids, similarity_scores = recommend_func(
            movie, top_n=10, min_rating=min_rating, genres=genres, year_range=year_range, mood=mood
        )
        
        recommended_movie_posters = []
        movie_details = []
        
        for i, movie_id in enumerate(recommended_movie_ids):
            poster = fetch_poster(movie_id)
            details = fetch_movie_details(movie_id)
            details['similarity_score'] = round(similarity_scores[i] * 100, 1)  # Convert similarity to percentage
            recommended_movie_posters.append(poster)
            movie_details.append(details)
        
        return recommended_movie_names, recommended_movie_posters, movie_details
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return [], [], []
    
# Define mood emoji mapping
mood_emoji = {
    'uplifting': '🌟',
    'positive': '😊',
    'neutral': '😐',
    'negative': '😟',
    'dark': '🖤',
    'action': '💥',
    'comedy': '😂',
    'romance': '❤️',
    'drama': '😭',
    'horror': '👻',
    'calm': '🌈',
    'fantasy': '🦄'
}

# Main function
def main():
    st.markdown('<h1 class="main-header">🎬 Movie Recommender System</h1>', unsafe_allow_html=True)
    
    # Load data
    movies, similarity, genres, moods_dict, _ = load_data()
    if movies is None:
        st.error("Failed to load necessary data. Please check the data files.")
        return
    
    # Create tabs for different recommendation methods
    tab1, tab2 = st.tabs(["Movie-Based Recommendations", "Mood-Based Recommendations"])
    
    with tab1:
        # Sidebar for filters
        with st.sidebar:
            st.header("About")
            st.info("""
            This application recommends movies similar to your selection.
            It uses machine learning to find patterns in movie features and suggest 
            titles you might enjoy based on your current favorite.
            """)
            
            st.header("Filters")
            min_rating = st.slider("Minimum Rating", 0.0, 10.0, 0.0, 0.5)
            
            # Dynamic genre list based on loaded data
            genre_filter = st.multiselect("Filter by Genre", sorted(genres))
            
            # Year range filter
            current_year = datetime.now().year
            year_range = st.slider("Release Year Range", 1900, current_year, (1900, current_year))
            
            # Mood filter
            mood_filter = st.selectbox("Movie Mood", ["Any"] + list(moods_dict.keys()))
            
            if st.button("Clear Filters"):
                st.session_state.min_rating = 0.0
                st.session_state.genre_filter = []
                st.session_state.year_range = (1900, current_year)
                st.session_state.mood_filter = "Any"
                st.experimental_rerun()
        
        # Search box with autocomplete dropdown
        col1, col2 = st.columns([3, 1])
        with col1:
            movie_list = movies['title'].values
            selected_movie = st.selectbox(
                "Search for a movie you like:",
                movie_list
            )
        with col2:
            recommend_button = st.button('Get Recommendations')
        
        # Show selected movie details
        if selected_movie:
            try:
                movie_data = movies[movies['title'] == selected_movie].iloc[0]
                movie_id = movie_data.movie_id
                
                poster_col, details_col = st.columns([1, 3])
                with poster_col:
                    poster_url = fetch_poster(movie_id)
                    st.image(poster_url, width=250)
                
                with details_col:
                    details = fetch_movie_details(movie_id)
                    st.subheader(f"{selected_movie}")
                    
                    # Year info
                    release_year = "Unknown"
                    if 'release_date' in details and details['release_date']:
                        release_year = details['release_date'].split('-')[0]
                    st.caption(f"Released: {release_year}")
                    
                    # Genre info
                    if 'genres' in details:
                        st.write(f"**Genres:** {details.get('genres', 'Not available')}")
                    
                    # Rating info
                    if 'rating' in details:
                        st.write(f"**Rating:** ⭐ {details.get('rating', 'N/A')}/10")
                    
                    # Mood info (from our analysis)
                    if 'mood' in movie_data:
                        mood = movie_data['mood']
                        st.write(f"**Mood:** {mood_emoji.get(mood, '🎭')} {mood.title()}")
                    
                    # Overview
                    if 'overview' in details:
                        st.write(f"**Overview:** {details.get('overview', 'No description available.')}")
                    
            except Exception as e:
                st.error(f"Error loading movie details: {e}")
        
        # Recommendations section
        if recommend_button:
            with st.spinner('Finding movies you might enjoy...'):
                # Process filters
                filter_mood = None if mood_filter == "Any" else mood_filter
                
                recommended_movie_names, recommended_movie_posters, movie_details = recommend(
                    selected_movie, 
                    min_rating=min_rating,
                    genres=genre_filter,
                    year_range=year_range,
                    mood=filter_mood
                )
                
                if not recommended_movie_names:
                    st.warning("No recommendations found with these filters. Try adjusting your criteria.")
                else:
                    st.markdown('<h2 class="recommendation-header">Recommended Movies For You</h2>', unsafe_allow_html=True)
                    
                    # Create tabs for different view options
                    view_tab1, view_tab2 = st.tabs(["Grid View", "Detailed View"])
                    
                    # Grid view
                    with view_tab1:
                        # Show recommendations in rows of 5
                        for i in range(0, len(recommended_movie_names), 5):
                            cols = st.columns(5)
                            for j in range(5):
                                if i+j < len(recommended_movie_names):
                                    with cols[j]:
                                        st.image(recommended_movie_posters[i+j], width=150)
                                        st.markdown(f'<p class="movie-title">{recommended_movie_names[i+j]}</p>', unsafe_allow_html=True)
                                        st.caption(f"Match: {movie_details[i+j]['similarity_score']}%")
                    
                    # Detailed view
                    with view_tab2:
                        for i in range(min(10, len(recommended_movie_names))):
                            with st.expander(f"{i+1}. {recommended_movie_names[i]} - {movie_details[i]['similarity_score']}% match"):
                                col1, col2 = st.columns([1, 3])
                                
                                with col1:
                                    st.image(recommended_movie_posters[i], width=200)
                                
                                with col2:
                                    st.subheader(recommended_movie_names[i])
                                    st.write(f"**Release Date:** {movie_details[i].get('release_date', 'N/A')}")
                                    st.write(f"**Genres:** {movie_details[i].get('genres', 'N/A')}")
                                    st.write(f"**Rating:** ⭐ {movie_details[i].get('rating', 'N/A')}/10")
                                    st.write(f"**Similarity Score:** {movie_details[i]['similarity_score']}%")
                                    st.write("**Overview:**")
                                    st.write(movie_details[i].get('overview', 'No description available'))

    # Mood-based recommendations tab
    with tab2:
        st.header("Discover Movies Based on Your Mood")
        st.write("Select a mood to find movies that match how you're feeling today:")
        
        # Create a grid of mood cards
        mood_cols = st.columns(4)
        
        # List of available moods
        available_moods = list(moods_dict.keys())
        
        # Create mood selection cards
        selected_mood = None
        for i, mood in enumerate(available_moods):
            with mood_cols[i % 4]:
                # Create a clickable card for each mood
                mood_card = st.container()
                with mood_card:
                    st.markdown(f"""
                    <div class="mood-card" id="mood-{mood}">
                        <div class="mood-emoji">{mood_emoji.get(mood, '🎭')}</div>
                        <h3>{mood.title()}</h3>
                        <p>{moods_dict.get(mood, '')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Select {mood.title()}", key=f"mood-btn-{mood}"):
                        selected_mood = mood
        
        # If a mood is selected, show random movies from that mood
        if selected_mood:
            st.subheader(f"{mood_emoji.get(selected_mood, '🎭')} {selected_mood.title()} Movies")
            
            # Get movies matching the selected mood
            filtered_movies = movies[movies['mood'] == selected_mood]
            
            # If no direct mood match, try mood categories
            if len(filtered_movies) < 5:
                for idx, row in movies.iterrows():
                    if selected_mood in row['mood_categories']:
                        if idx not in filtered_movies.index:
                            filtered_movies = pd.concat([filtered_movies, movies.loc[[idx]]])
            
            # Fallback to random selection with some mood similarity
            if len(filtered_movies) < 5:
                st.info(f"Not many movies directly match the {selected_mood} mood. Here are some that might be close:")
                # Get a larger random sample
                filtered_movies = movies.sample(min(20, len(movies)))
            
            # Sample 10 movies from the filtered list
            if len(filtered_movies) > 10:
                mood_movies = filtered_movies.sample(10)
            else:
                mood_movies = filtered_movies
            
            # Display movies in a grid
            cols = st.columns(5)
            for i, (_, movie) in enumerate(mood_movies.iterrows()):
                col_idx = i % 5
                with cols[col_idx]:
                    poster = fetch_poster(movie['movie_id'])
                    st.image(poster, width=150)
                    st.markdown(f'<p class="movie-title">{movie["title"]}</p>', unsafe_allow_html=True)
                    
                    # Option to see similar movies
                    if st.button(f"See similar", key=f"similar-{i}"):
                        st.session_state.selected_movie = movie['title']
                        st.session_state.show_recommendations = True
                        st.experimental_rerun()
            
            # Additional information about the mood
            st.write(f"**About {selected_mood.title()} Movies:**")
            st.write(moods_dict.get(selected_mood, "Movies that match your current mood."))

# Run the application
if __name__ == "__main__":
    main()
'''