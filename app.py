import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests

# Initialize session state defaults
if "username" not in st.session_state:
    st.session_state.username = ""
if "refresh" not in st.session_state:
    st.session_state.refresh = False
if "search_terms" not in st.session_state:
    st.session_state.search_terms = []

# Load books dataset
books = pd.read_csv('data/Books.csv', sep=',', encoding='utf-8', on_bad_lines='skip', low_memory=False)
books.columns = books.columns.str.strip()
desired_columns = ['ISBN', 'Book-Title', 'Book-Author', 'Image-URL-M', 'Publisher', 'Genre']
books = books[[col for col in desired_columns if col in books.columns]].dropna().drop_duplicates()

# Fix encoding issues in book titles and authors
books['Book-Title'] = books['Book-Title'].str.encode('utf-8', errors='ignore').str.decode('utf-8')
books['Book-Author'] = books['Book-Author'].str.encode('utf-8', errors='ignore').str.decode('utf-8')

# Load ratings dataset
ratings = pd.read_csv('data/Ratings.csv', sep=',', encoding='utf-8', on_bad_lines='skip', low_memory=False)
ratings.columns = ratings.columns.str.strip()
rating_avg = ratings.groupby('ISBN')['Book-Rating'].mean().reset_index()
rating_avg.columns = ['ISBN', 'Average-Rating']

# Merge ratings with books
books = pd.merge(books, rating_avg, on='ISBN', how='left')

# Create TF-IDF text using only Book Titles
@st.cache_resource
def load_tfidf(title_series):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(title_series)
    return tfidf, tfidf_matrix

tfidf, tfidf_matrix = load_tfidf(books['Book-Title'])

# Create user_data directory and history file
os.makedirs('user_data', exist_ok=True)
history_path = 'user_data/search_history.csv'
if not os.path.exists(history_path):
    pd.DataFrame(columns=['search']).to_csv(history_path, index=False)

def load_search_history():
    return pd.read_csv(history_path)

def save_search(term):
    df = load_search_history()
    df = pd.concat([df, pd.DataFrame([[term]], columns=['search'])])
    df.to_csv(history_path, index=False)

def clear_search_history():
    pd.DataFrame(columns=['search']).to_csv(history_path, index=False)

def get_search_terms():
    return load_search_history()['search'].tolist()

@st.cache_data(show_spinner=False)
def get_google_books_image(title, author=None):
    try:
        query = f"intitle:{title}"
        if author:
            query += f"+inauthor:{author}"
        url = f"https://www.googleapis.com/books/v1/volumes?q={query}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if "items" in data and len(data["items"]) > 0:
                for item in data["items"]:
                    image_links = item['volumeInfo'].get('imageLinks', {})
                    image_url = image_links.get("thumbnail") or image_links.get("smallThumbnail")
                    if image_url:
                        return image_url
    except:
        pass
    return None

def resolve_image_url(title, author, default_url):
    api_img = get_google_books_image(title, author)
    if api_img:
        return api_img
    if pd.notna(default_url) and str(default_url).startswith("http"):
        return default_url
    return 'https://via.placeholder.com/80x100?text=No+Image'

def recommend_books(search_terms, excluded_titles):
    if not search_terms:
        return pd.DataFrame()
    sim_scores = []
    for term in search_terms:
        term_vec = tfidf.transform([term])
        sim = cosine_similarity(term_vec, tfidf_matrix).flatten()
        sim_scores.append(sim)

    weights = np.linspace(1, 2, num=len(sim_scores))
    combined_scores = np.average(sim_scores, axis=0, weights=weights)

    top_indices = combined_scores.argsort()[::-1]
    recommendations = books.iloc[top_indices]
    recommendations = recommendations[~recommendations['Book-Title'].isin(excluded_titles)]
    recommendations = recommendations[~recommendations['Book-Title'].str.lower().isin([s.lower() for s in search_terms])]
    recommendations = recommendations.drop_duplicates(subset='Book-Title').head(50)

    diverse_recs = recommendations.groupby('Book-Author', group_keys=False).apply(lambda x: x.sample(1, random_state=42)).reset_index(drop=True)

    if len(diverse_recs) > 5:
        diverse_recs = diverse_recs.sample(5, random_state=42)
    return diverse_recs

def trending_books():
    top_trending = books.sort_values(by='Average-Rating', ascending=False).dropna(subset=['Average-Rating'])
    top_trending = top_trending.drop_duplicates(subset='Book-Title').head(20)
    return top_trending

st.title("Book Recommendation System")

AUTHORIZED_USERNAME = "user1"
if st.session_state.get("username") == AUTHORIZED_USERNAME:
    st.sidebar.write("👤 Logged in")
    if st.sidebar.button("Logout"):
        st.session_state.username = ""
        st.session_state.search_terms = []
        st.rerun()
else:
    username_input = st.sidebar.text_input("Enter username")
    if st.sidebar.button("Login"):
        if username_input.strip().lower() == AUTHORIZED_USERNAME:
            st.session_state.username = AUTHORIZED_USERNAME
            st.success("Logged in successfully.")
            st.session_state.search_terms = get_search_terms()
            st.rerun()
        else:
            st.error("Invalid username")

if st.session_state.get("username") == AUTHORIZED_USERNAME:
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Your Recommendations", "Top Rated Books", "Search History", "More Recommendations"])

    with tab1:
        search_input = st.text_input("Search for books using book title keywords")
        searched_titles = []
        if st.button("Search"):
            if search_input.strip():
                save_search(search_input.strip())
                st.session_state.search_terms.append(search_input.strip())
                search_query = search_input.strip().lower()
                results = books[books['Book-Title'].str.lower().str.contains(search_query, na=False)].drop_duplicates(subset='Book-Title').head(5)
                if results.empty:
                    st.warning("Book not found.")
                else:
                    searched_titles = results['Book-Title'].tolist()
                    st.write("### Search Results:")
                    for _, row in results.iterrows():
                        image_url = resolve_image_url(row['Book-Title'], row['Book-Author'], row['Image-URL-M'])
                        st.markdown(f"""
                            <div style='display: flex; gap: 1rem; align-items: center;'>
                                <img src=\"{image_url}\" width=\"80\"/>
                                <div>
                                    <strong>Title:</strong> {row['Book-Title']}<br>
                                    <strong>Author:</strong> {row['Book-Author']}<br>
                                    <strong>Publisher:</strong> {row['Publisher']}<br>
                                    <strong>Genre:</strong> {row['Genre']}<br>
                                    <strong>Rating:</strong> {row['Average-Rating']:.2f}
                                </div>
                            </div>
                            <hr>
                        """, unsafe_allow_html=True)

        if len(get_search_terms()) >= 4:
            st.write("### Recommended For You:")
            past_search_terms = get_search_terms()[-4:]

            you_may_like = recommend_books(past_search_terms, searched_titles)
            you_may_like = you_may_like.drop_duplicates(subset='Book-Title')

            if you_may_like.empty:
                st.info("No recommendations available based on your search history.")
            for _, row in you_may_like.iterrows():
                image_url = resolve_image_url(row['Book-Title'], row['Book-Author'], row['Image-URL-M'])
                st.markdown(f"""
                    <div style='display: flex; gap: 1rem; align-items: center;'>
                        <img src=\"{image_url}\" width=\"80\"/>
                        <div>
                            <strong>Title:</strong> {row['Book-Title']}<br>
                            <strong>Author:</strong> {row['Book-Author']}<br>
                            <strong>Publisher:</strong> {row['Publisher']}<br>
                            <strong>Genre:</strong> {row['Genre']}<br>
                            <strong>Rating:</strong> {row['Average-Rating']:.2f}
                        </div>
                    </div>
                    <hr>
                """, unsafe_allow_html=True)

    with tab2:
        st.write("### Top Rated Books")
        trending = trending_books()
        for _, row in trending.iterrows():
            image_url = resolve_image_url(row['Book-Title'], row['Book-Author'], row['Image-URL-M'])
            st.markdown(f"""
                <div style='display: flex; gap: 1rem; align-items: center;'>
                    <img src=\"{image_url}\" width=\"80\"/>
                    <div>
                        <strong>Title:</strong> {row['Book-Title']}<br>
                        <strong>Author:</strong> {row['Book-Author']}<br>
                        <strong>Publisher:</strong> {row['Publisher']}<br>
                        <strong>Genre:</strong> {row['Genre']}<br>
                        <strong>Rating:</strong> {row['Average-Rating']:.2f}
                    </div>
                </div>
                <hr>
            """, unsafe_allow_html=True)

    with tab3:
        st.write("### 🔎 Your Search History")
        history_terms = get_search_terms()
        if not history_terms:
            st.info("No search history found.")
        else:
            for i, term in enumerate(history_terms[::-1], 1):
                st.markdown(f"**{i}.** {term}")

        if st.button("Clear Search History"):
            clear_search_history()
            st.session_state.search_terms = []
            st.success("Search history cleared.")
            st.rerun()

    with tab4:
        st.write("### Discover More Based on Your Interests")

        history_terms = get_search_terms()
        if not history_terms:
            st.info("Search history is empty. Search some books to get recommendations.")
        else:
            matched_books = books[books['Book-Title'].str.lower().apply(
                lambda title: any(term.lower() in title for term in history_terms)
            )]

            # Extract unique values
            authors = matched_books['Book-Author'].dropna().unique()
            publishers = matched_books['Publisher'].dropna().unique()
            genres = matched_books['Genre'].dropna().unique()

            filter_option = st.selectbox("Choose a recommendation type:", ["Author", "Publisher", "Genre"])

            selected_value = None
            recs = pd.DataFrame()

            if filter_option == "Author":
                selected_value = st.selectbox("Select an author:", sorted(authors))
                if selected_value:
                    recs = books[books['Book-Author'] == selected_value]

            elif filter_option == "Publisher":
                selected_value = st.selectbox("Select a publisher:", sorted(publishers))
                if selected_value:
                    recs = books[books['Publisher'] == selected_value]

            elif filter_option == "Genre":
                selected_value = st.selectbox("Select a genre:", sorted(genres))
                if selected_value:
                    recs = books[books['Genre'] == selected_value]

            # Show top 10 unique recommendations
            if not recs.empty:
                recs = recs.drop_duplicates(subset='Book-Title')
                if len(recs) > 10:
                    recs = recs.sample(10, random_state=42)

            if recs.empty and selected_value:
                st.warning("No recommendations found for the selected option.")
            for _, row in recs.iterrows():
                image_url = resolve_image_url(row['Book-Title'], row['Book-Author'], row['Image-URL-M'])
                st.markdown(f"""
                    <div style='display: flex; gap: 1rem; align-items: center;'>
                        <img src="{image_url}" width="80"/>
                        <div>
                            <strong>Title:</strong> {row['Book-Title']}<br>
                            <strong>Author:</strong> {row['Book-Author']}<br>
                            <strong>Publisher:</strong> {row['Publisher']}<br>
                            <strong>Genre:</strong> {row['Genre']}<br>
                            <strong>Rating:</strong> {row['Average-Rating']:.2f}
                        </div>
                    </div>
                    <hr>
                """, unsafe_allow_html=True)


else:
    st.info("Please log in using username")
