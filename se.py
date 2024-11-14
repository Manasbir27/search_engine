import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
class TimelineVisualizer:
    def __init__(self, books_df, movies_df):
        self.books_df = books_df
        self.movies_df = movies_df
        
    def create_franchise_timeline(self, franchise_name):
        """Create a timeline visualization for a movie franchise."""
        # Filter movies by franchise name (case-insensitive partial match)
        franchise_movies = self.movies_df[
            self.movies_df['title'].str.contains(franchise_name, case=False, na=False)
        ].copy()
        
        if franchise_movies.empty:
            return "No movies found for this franchise."
            
        # Sort by release date
        franchise_movies['parsed_date'] = pd.to_datetime(franchise_movies['release_date'])
        franchise_movies = franchise_movies.sort_values('parsed_date')
        
        # Create the visualization
        plt.figure(figsize=(15, 8))
        
        # Plot points for each movie
        plt.scatter(franchise_movies['parsed_date'], 
                   range(len(franchise_movies)), 
                   s=100, 
                   color='blue')
        
        # Add movie titles as labels
        for idx, movie in franchise_movies.iterrows():
            plt.annotate(
                f"{movie['title']}\n({movie['vote_average']:.1f}/10)",
                (movie['parsed_date'], franchise_movies.index.get_loc(idx)),
                xytext=(10, 0), 
                textcoords='offset points',
                va='center'
            )
            
        plt.yticks([])  # Hide y-axis ticks
        plt.xlabel('Release Year')
        plt.title(f'Timeline: {franchise_name} Franchise')
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save and show the plot
        plt.savefig('franchise_timeline.png')
        plt.show()  # Add this line to display the plot
        plt.close()
        
        return "Timeline has been created and saved as 'franchise_timeline.png'"
        
    def create_author_timeline(self, author_name):
        """Create a timeline visualization for an author's books."""
        # Create more flexible author name matching
        author_variations = [
            author_name.lower(),
            author_name.upper(),
            author_name.title(),
            # For J.K. Rowling specifically
            'J.K. Rowling',
            'JK Rowling',
            'Rowling, J.K.',
            'Rowling, J. K.',
            'J. K. Rowling'
        ]
        
        # Filter books using any of the author name variations
        author_books = self.books_df[
            self.books_df['authors'].str.contains('|'.join(author_variations), 
                                                case=False, 
                                                na=False,
                                                regex=True)
        ].copy()
        
        if author_books.empty:
            # Debug information
            print(f"No books found. Available author formats in database:")
            sample_authors = self.books_df['authors'].dropna().sample(min(5, len(self.books_df))).tolist()
            print("\n".join(sample_authors))
            return "No books found for this author. Please check the author name format."
            
        # Clean and parse dates
        author_books['parsed_date'] = pd.to_datetime(author_books['publication_date'], errors='coerce')
        
        # Remove entries with invalid dates
        author_books = author_books.dropna(subset=['parsed_date'])
        
        if author_books.empty:
            return "No books found with valid publication dates for this author."
        
        # Sort by publication date
        author_books = author_books.sort_values('parsed_date')
        
        # Rest of the visualization code remains the same
        try:
            plt.figure(figsize=(15, 10))
            
            # Create timeline plot
            plt.subplot(2, 1, 1)
            scatter = plt.scatter(author_books['parsed_date'], 
                                range(len(author_books)), 
                                s=100, 
                                c=author_books['average_rating'],
                                cmap='RdYlGn',
                                vmin=1, vmax=5)
            
            # Add colorbar for ratings
            plt.colorbar(scatter, label='Average Rating')
            
            # Add book titles as labels
            for idx, book in author_books.iterrows():
                label_text = f"{book['title']}\n({book['average_rating']:.1f}/5)"
                plt.annotate(
                    label_text,
                    (book['parsed_date'], author_books.index.get_loc(idx)),
                    xytext=(10, 0), 
                    textcoords='offset points',
                    va='center',
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
                )
            
            plt.yticks([])
            plt.xlabel('Publication Year')
            plt.title(f'Timeline: Books by {author_name}')
            plt.grid(True, alpha=0.3)
            
            # Create the ratings distribution subplot
            plt.subplot(2, 1, 2)
            rating_counts = pd.cut(author_books['average_rating'], 
                                bins=np.arange(1, 6, 0.5),
                                include_lowest=True).value_counts()
            rating_counts.plot(kind='bar')
            plt.xlabel('Rating Range')
            plt.ylabel('Number of Books')
            plt.title('Distribution of Book Ratings')
            plt.xticks(rotation=45)
            
            # Calculate summary statistics
            summary_stats = {
                'Total Books': len(author_books),
                'Average Rating': author_books['average_rating'].mean(),
                'Highest Rated': author_books.loc[author_books['average_rating'].idxmax(), 'title'],
                'Most Recent': author_books.loc[author_books['parsed_date'].idxmax(), 'title'],
                'First Published': author_books.loc[author_books['parsed_date'].idxmin(), 'title'],
                'Publication Span': f"{author_books['parsed_date'].min().year} - {author_books['parsed_date'].max().year}"
            }
            
            plt.figtext(0.02, 0.02, 
                    '\n'.join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" 
                                for k, v in summary_stats.items()]),
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig('author_timeline.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()
            
            return {
                "message": "Timeline has been created and saved as 'author_timeline.png'",
                "stats": summary_stats,
                "books": author_books[['title', 'average_rating', 'parsed_date']].to_dict('records')
            }
            
        except Exception as e:
            plt.close()
            return f"Error creating timeline visualization: {str(e)}"

    def create_genre_graph(self, media_type='movies'):
        """Create a graph showing genre distribution over time."""
        if media_type.lower() == 'movies':
            # Process movie genres
            genre_by_year = defaultdict(lambda: defaultdict(int))
            
            for _, movie in self.movies_df.iterrows():
                if pd.notna(movie['release_date']) and movie['genres']:
                    year = pd.to_datetime(movie['release_date']).year
                    for genre in movie['genres']:
                        if isinstance(genre, dict):
                            genre_by_year[year][genre['name']] += 1
                            
            # Convert to DataFrame for plotting
            years = sorted(genre_by_year.keys())
            genres = set()
            for year_data in genre_by_year.values():
                genres.update(year_data.keys())
            
            data = []
            for year in years:
                for genre in genres:
                    data.append({
                        'Year': year,
                        'Genre': genre,
                        'Count': genre_by_year[year][genre]
                    })
            
            df = pd.DataFrame(data)
            
        else:  # books
            # For books, we'll need to process genres differently
            genre_by_year = defaultdict(lambda: defaultdict(int))
            
            for _, book in self.books_df.iterrows():
                if pd.notna(book['publication_date']):
                    year = pd.to_datetime(book['publication_date']).year
                    genre = "General"  # Replace with actual genre processing
                    genre_by_year[year][genre] += 1
                    
            df = pd.DataFrame([(year, genre, count) 
                             for year, genres in genre_by_year.items() 
                             for genre, count in genres.items()],
                            columns=['Year', 'Genre', 'Count'])
        
        # Create the visualization
        plt.figure(figsize=(15, 8))
        
        # Create a line plot for each genre
        for genre in df['Genre'].unique():
            genre_data = df[df['Genre'] == genre]
            plt.plot(genre_data['Year'], genre_data['Count'], 
                    label=genre, marker='o')
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlabel('Year')
        plt.ylabel('Number of Titles')
        plt.title(f'Genre Distribution Over Time ({media_type.title()})')
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save and show the plot
        plt.savefig('genre_distribution.png')
        plt.show()  # Add this line to display the plot
        plt.close()
        
        return "Genre distribution graph has been created and saved as 'genre_distribution.png'"

class UserPreferences:
    def __init__(self):
        # Define user profiles with their preferences
        self.users = {
            'manas': {
                'favorite_genres': ['Science Fiction', 'Action', 'Thriller'],
                'min_rating': 7.0,
                'preferred_years': range(2010, 2025),
                'favorite_authors': ['Andy Weir', 'Blake Crouch', 'Stephen King'],
                'preferred_book_genres': ['Science Fiction', 'Mystery', 'Thriller'],
                'reading_level': 'advanced',
                'preferred_movie_length': range(90, 150),  # in minutes
                'preferred_pace': 'fast',
                'themes': ['space', 'technology', 'suspense', 'mystery'],
                'avoid_genres': ['Romance', 'Musical']
            },
            'arin': {
                'favorite_genres': ['Drama', 'Comedy', 'Romance'],
                'min_rating': 6.5,
                'preferred_years': range(2000, 2025),
                'favorite_authors': ['John Green', 'Rainbow Rowell', 'Jane Austen'],
                'preferred_book_genres': ['Young Adult', 'Contemporary', 'Romance'],
                'reading_level': 'intermediate',
                'preferred_movie_length': range(90, 135),  # in minutes
                'themes': ['relationships', 'coming-of-age', 'humor', 'life-lessons'],
                'avoid_genres': ['Horror', 'War']
            }
        }
class SearchEngine:
    def __init__(self):
        print("Initializing search engine...")
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            print("Warning: NLTK data download failed. Using basic preprocessing.")

        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()

        self.vectorizer = TfidfVectorizer(
            preprocessor=self.preprocess_text,
            stop_words='english',
            max_features=10000
        )

        # Load datasets
        print("Loading datasets...")
        self.books_df = pd.read_csv('books.csv', on_bad_lines='skip')
        self.movies_df = pd.read_csv('movies.csv', on_bad_lines='skip')

        # Convert ratings to numeric and handle NaN values
        self.books_df['average_rating'] = pd.to_numeric(self.books_df['average_rating'], errors='coerce')
        self.movies_df['vote_average'] = pd.to_numeric(self.movies_df['vote_average'], errors='coerce')

        # Convert budget to numeric and handle NaN values
        self.movies_df['budget'] = pd.to_numeric(self.movies_df['budget'], errors='coerce')

        # Parse dates
        self.books_df['parsed_date'] = self.books_df['publication_date'].apply(self.parse_date)
        self.movies_df['parsed_date'] = pd.to_datetime(self.movies_df['release_date'], errors='coerce')

        # Create genre lists from genre string
        self.movies_df['genres'] = self.movies_df['genres'].fillna('[]').apply(self.parse_genres)

        print("Creating search indices...")
        self.create_search_features()
        self.create_search_index()
        print("Search engine is ready!")

    def parse_genres(self, genres_str):
        """Parse genres from string representation."""
        try:
            if isinstance(genres_str, str):
                return eval(genres_str)
            return []
        except:
            return []

    def parse_date(self, date_str):
        """Convert various date formats to datetime object."""
        if pd.isna(date_str):
            return None

        try:
            # Try different date formats
            date_formats = [
                '%m-%d-%Y',
                '%Y-%m-%d',
                '%m/%d/%Y',
                '%Y/%m/%d',
                '%d-%m-%Y',
                '%Y',
                '%m-%Y',
                '%Y-%m'
            ]

            for fmt in date_formats:
                try:
                    return pd.to_datetime(date_str, format=fmt)
                except:
                    continue

            # If specific formats fail, try pandas default parser
            return pd.to_datetime(date_str)
        except:
            return None

    def preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = word_tokenize(text)
        return ' '.join([self.stemmer.stem(token) for token in tokens
                        if token not in self.stop_words and token.isalnum()])

    def create_search_features(self):
        # Create search features for books
        self.books_df['search_features'] = (
            self.books_df['title'].fillna('') + ' ' +
            self.books_df['authors'].fillna('') + ' ' +
            'published in ' + self.books_df['publication_date'].fillna('') + ' ' +
            'rating ' + self.books_df['average_rating'].astype(str) + ' ' +
            'pages ' + self.books_df['num_pages'].astype(str)
        )

        # Create search features for movies
        self.movies_df['search_features'] = (
            self.movies_df['title'].fillna('') + ' ' +
            self.movies_df['overview'].fillna('') + ' ' +
            self.movies_df['genres'].apply(lambda x: ' '.join([item['name'] for item in x if isinstance(item, dict)])) + ' ' +
            'released in ' + self.movies_df['release_date'].fillna('') + ' ' +
            'rating ' + self.movies_df['vote_average'].astype(str) + ' ' +
            'duration ' + self.movies_df['runtime'].astype(str)
        )
    def get_personalized_recommendations(self, username):
        """Get personalized recommendations for a specific user."""
        if username.lower() not in self.user_preferences.users:
            return f"User {username} not found. Available users: {', '.join(self.user_preferences.users.keys())}"

        user_prefs = self.user_preferences.users[username.lower()]
        results = []

        # Movie recommendations
        movie_matches = []
        for _, movie in self.movies_df.iterrows():
            movie_genres = [g['name'] for g in movie['genres'] if isinstance(g, dict)]
            
            # Check if movie matches user preferences
            if (any(genre in movie_genres for genre in user_prefs['favorite_genres']) and
                not any(genre in movie_genres for genre in user_prefs['avoid_genres']) and
                pd.notna(movie['vote_average']) and movie['vote_average'] >= user_prefs['min_rating'] and
                pd.notna(movie['runtime']) and movie['runtime'] in user_prefs['preferred_movie_length'] and
                pd.notna(movie['release_date']) and 
                int(movie['release_date'][:4]) in user_prefs['preferred_years']):
                
                # Check for preferred themes in overview
                theme_match = any(theme.lower() in str(movie['overview']).lower() 
                                for theme in user_prefs['themes'])
                
                if theme_match:
                    movie_matches.append({
                        'type': 'movie',
                        'title': movie['title'],
                        'rating': f"{movie['vote_average']:.1f}/10",
                        'release_date': movie['release_date'],
                        'genres': movie_genres,
                        'overview': movie['overview'][:200] + "..." if len(str(movie['overview'])) > 200 else str(movie['overview'])
                    })

        # Book recommendations
        book_matches = []
        for _, book in self.books_df.iterrows():
            # Check if book matches user preferences
            if (pd.notna(book['average_rating']) and 
                book['average_rating'] >= (user_prefs['min_rating'] * 0.5) and  # Convert 10-point scale to 5-point
                any(author in str(book['authors']) for author in user_prefs['favorite_authors'])):
                
                book_matches.append({
                    'type': 'book',
                    'title': book['title'],
                    'author': book['authors'],
                    'rating': f"{book['average_rating']:.1f}/5",
                    'publication_date': book['publication_date']
                })

        # Sort and combine recommendations
        movie_matches.sort(key=lambda x: float(x['rating'].split('/')[0]), reverse=True)
        book_matches.sort(key=lambda x: float(x['rating'].split('/')[0]), reverse=True)

        results.extend(movie_matches[:3])  # Top 3 movie recommendations
        results.extend(book_matches[:2])   # Top 2 book recommendations

        return results

           

    def create_search_index(self):
        combined_features = pd.concat([
            self.books_df['search_features'],
            self.movies_df['search_features']
        ]).fillna('')

        self.vectorizer.fit(combined_features)
        self.tfidf_matrix_books = self.vectorizer.transform(self.books_df['search_features'])
        self.tfidf_matrix_movies = self.vectorizer.transform(self.movies_df['search_features'])

    def search_books_by_date(self, query):
        """Search for books based on date criteria."""
        date_patterns = {
            'after': r'(?:released |published )?after (\d{4})',
            'before': r'(?:released |published )?before (\d{4})',
            'in': r'(?:released |published )?in (\d{4})',
            'between': r'(?:released |published )?between (\d{4}) and (\d{4})'
        }

        # Find matching criteria
        results = []
        year = None
        end_year = None

        for criteria, pattern in date_patterns.items():
            match = re.search(pattern, query.lower())
            if match:
                if criteria == 'between':
                    year, end_year = map(int, match.groups())
                else:
                    year = int(match.group(1))
                break

        if year:
            filtered_df = self.books_df.copy()

            if criteria == 'after':
                filtered_df = filtered_df[filtered_df['parsed_date'] >= pd.to_datetime(f'{year}-01-01')]
            elif criteria == 'before':
                filtered_df = filtered_df[filtered_df['parsed_date'] < pd.to_datetime(f'{year}-01-01')]
            elif criteria == 'in':
                filtered_df = filtered_df[filtered_df['parsed_date'].dt.year == year]
            elif criteria == 'between':
                filtered_df = filtered_df[
                    (filtered_df['parsed_date'] >= pd.to_datetime(f'{year}-01-01')) &
                    (filtered_df['parsed_date'] <= pd.to_datetime(f'{end_year}-12-31'))
                ]

            # Sort by date descending for 'after', ascending for 'before'
            filtered_df = filtered_df.sort_values(
                'parsed_date',
                ascending=criteria == 'before'
            )

            # Get top 5 results
            for _, book in filtered_df.head(5).iterrows():
                if pd.notna(book['parsed_date']):  # Only include books with valid dates
                    results.append({
                        'type': 'book',
                        'title': book['title'],
                        'author': book['authors'],
                        'rating': f"{book['average_rating']:.1f}/5" if pd.notna(book['average_rating']) else 'N/A',
                        'publication_date': book['parsed_date'].strftime('%Y-%m-%d'),
                        'relevance': 1.0
                    })

            return results if results else "No books found matching the date criteria."

        return None

    def search_by_genre_and_rating(self, genre, min_rating):
        """Search for movies by genre and minimum rating."""
        matching_movies = []
        for _, movie in self.movies_df.iterrows():
            genres = [g['name'].lower() for g in movie['genres'] if isinstance(g, dict)]
            if (genre.lower() in genres and
                pd.notna(movie['vote_average']) and
                movie['vote_average'] >= min_rating):
                matching_movies.append({
                    'type': 'movie',
                    'title': movie['title'],
                    'rating': f"{movie['vote_average']:.1f}/10",
                    'release_date': movie['release_date'],
                    'budget': f"${movie['budget']:,.0f}" if pd.notna(movie['budget']) and movie['budget'] > 0 else 'Unknown',
                    'overview': movie['overview'][:200] + "..." if len(str(movie['overview'])) > 200 else str(movie['overview'])
                })

        # Sort by rating descending
        matching_movies.sort(key=lambda x: float(x['rating'].split('/')[0]), reverse=True)
        return matching_movies[:5] if matching_movies else "No movies found matching the criteria."

    def search_by_hit_and_latest(self, genre):
        """Search for hit and latest movies by genre."""
        matching_movies = []
        for _, movie in self.movies_df.iterrows():
            genres = [g['name'].lower() for g in movie['genres'] if isinstance(g, dict)]
            if genre.lower() in genres:
                if pd.notna(movie['vote_average']) and movie['vote_average'] >= 6.5 and movie['parsed_date'].year >= 2015:
                    matching_movies.append({
                        'type': 'movie',
                        'title': movie['title'],
                        'rating': f"{movie['vote_average']:.1f}/10",
                        'release_date': movie['parsed_date'].year,
                        'budget': f"${movie['budget']:,.0f}" if pd.notna(movie['budget']) and movie['budget'] > 0 else 'Unknown',
                        'overview': movie['overview'][:200] + "..." if len(str(movie['overview'])) > 200 else str(movie['overview'])
                    })

        # Sort by release date descending
        matching_movies.sort(key=lambda x: x['release_date'], reverse=True)
        return matching_movies[:5] if matching_movies else "No hit or latest movies found matching the genre."

    def sort_results(self, results, sort_by=None, reverse=False):
        """Generic sorting function for search results."""
        if not isinstance(results, list) or not results:
            return results
            
        if sort_by == 'budget':
            return self.sort_by_budget(results, reverse)
        elif sort_by == 'rating':
            return self.sort_by_rating(results, reverse)
        elif sort_by == 'date':
            return self.sort_by_date(results, reverse)
        elif sort_by == 'relevance':
            return self.sort_by_relevance(results, reverse)
        return results

    def sort_by_budget(self, results, reverse=False):
        """Sort results by budget."""
        def get_budget(item):
            if item.get('type') != 'movie' or 'budget' not in item:
                return -1
            budget_str = item['budget']
            if budget_str == 'Unknown':
                return -1
            try:
                return float(budget_str.replace('$', '').replace(',', ''))
            except:
                return -1
                
        return sorted(results, key=get_budget, reverse=reverse)

    def sort_by_rating(self, results, reverse=False):
        """Sort results by rating."""
        def get_rating(item):
            rating_str = item.get('rating', '0/0')
            try:
                value, scale = map(float, rating_str.split('/'))
                # Normalize ratings to 10-point scale
                return (value / scale) * 10
            except:
                return -1
                
        return sorted(results, key=get_rating, reverse=reverse)

    def sort_by_date(self, results, reverse=False):
        """Sort results by release/publication date."""
        def get_date(item):
            date_str = item.get('release_date') if item.get('type') == 'movie' else item.get('publication_date')
            if not date_str:
                return datetime.min
            try:
                return pd.to_datetime(date_str)
            except:
                return datetime.min
                
        return sorted(results, key=get_date, reverse=reverse)

    def sort_by_relevance(self, results, reverse=False):
        """Sort results by search relevance score."""
        return sorted(results, key=lambda x: float(x.get('relevance', 0)), reverse=reverse)

    def parse_query(self, query):
        query = query.lower()

        parsed = {
            'type': None,
            'attribute': None,
            'title': None,
            'similar': False,
            'genre': None,
            'hit_and_latest': False
        }

        # Detect question type
        if any(word in query for word in ['rating', 'rated', 'score']):
            parsed['attribute'] = 'rating'
        elif any(word in query for word in ['year', 'when', 'released', 'published']):
            parsed['attribute'] = 'year'
        elif any(word in query for word in ['overview', 'plot', 'about']):
            parsed['attribute'] = 'overview'
        elif 'budget' in query:
            parsed['attribute'] = 'budget'

        # Detect similarity request
        if any(word in query for word in ['similar', 'like']):
            parsed['similar'] = True

        # Extract title
        title_patterns = [
            r'of\s+([^?]+?)(?:\s+movie|\s+book|\s*\?|\s*$)',
            r'(?:movie|book|film)\s+([^?]+?)(?:\s*\?|\s*$)',
            r'about\s+([^?]+?)(?:\s*\?|\s*$)',
            r'"(.+?)"'
        ]

        for pattern in title_patterns:
            match = re.search(pattern, query)
            if match:
                parsed['title'] = match.group(1).strip()
                break

        # Detect media type
        if 'movie' in query or 'film' in query:
            parsed['type'] = 'movie'
        elif 'book' in query:
            parsed['type'] = 'book'

        # Detect "hit" and "latest" keywords
        if any(word in query for word in ['hit', 'latest']):
            parsed['hit_and_latest'] = True
            genre_match = re.search(r'(\w+)\s+(?:hit|latest)\s+movies?', query)
            if genre_match:
                parsed['genre'] = genre_match.group(1)

        return parsed

    def search(self, query):
        # Extract sorting criteria
        sort_by, reverse = self.parse_sort_criteria(query)
        
        # First check for date-based queries
        date_results = self.search_books_by_date(query)
        if date_results:
            return self.sort_results(date_results, sort_by, reverse)

        # Check for genre and rating queries
        genre_rating_match = re.search(r'(\w+)\s+movies?\s+with\s+rating\s+above\s+(\d+\.?\d*)', query.lower())
        if genre_rating_match:
            genre, min_rating = genre_rating_match.groups()
            results = self.search_by_genre_and_rating(genre, float(min_rating))
            return self.sort_results(results, sort_by, reverse)

        # Check for "hit" and "latest" queries
        hit_and_latest_match = re.search(r'(\w+)\s+(?:hit|latest)\s+movies?', query.lower())
        if hit_and_latest_match:
            genre = hit_and_latest_match.group(1)
            results = self.search_by_hit_and_latest(genre)
            return self.sort_results(results, sort_by, reverse)

        # Regular search
        parsed = self.parse_query(query)
        # Check for personalized recommendation requests
        user_recommendation_match = re.search(r'recommend\s+(?:some\s+)?(?:movies|books)\s+for\s+(\w+)', query.lower())
        if user_recommendation_match:
            username = user_recommendation_match.group(1)
            return self.get_personalized_recommendations(username)
        # Check for budget-specific queries
        budget_match = re.search(r'budget\s+(?:more|greater|higher|above|over)\s+than\s+(\d+(?:,\d+)*)', query.lower())
        if budget_match:
            min_budget = float(budget_match.group(1).replace(',', ''))
            results = self.search_by_budget(min_budget)
            return self.sort_results(results, 'budget', True)

        if parsed['title']:
            result = self.find_exact_match(parsed['title'], parsed)
            if result:
                return result

        # Regular search with TF-IDF
        results = self.perform_tfidf_search(query, parsed)
        return self.sort_results(results, sort_by, reverse)

    def search_by_budget(self, min_budget):
        """Search for movies with budget above specified amount."""
        results = []
        for _, movie in self.movies_df.iterrows():
            if pd.notna(movie['budget']) and movie['budget'] >= min_budget:
                results.append(self.format_movie_details(movie))
        return results

    def perform_tfidf_search(self, query, parsed):
        """Perform TF-IDF based search."""
        processed_query = self.preprocess_text(query)
        query_vector = self.vectorizer.transform([processed_query])
        results = []

        if parsed['type'] in [None, 'movie']:
            movie_scores = cosine_similarity(query_vector, self.tfidf_matrix_movies).flatten()
            top_movie_indices = np.argsort(movie_scores)[::-1][:5]
            
            for idx in top_movie_indices:
                if movie_scores[idx] > 0:
                    movie = self.movies_df.iloc[idx]
                    results.append(self.format_movie_details(movie) | {'relevance': movie_scores[idx]})

        if parsed['type'] in [None, 'book']:
            book_scores = cosine_similarity(query_vector, self.tfidf_matrix_books).flatten()
            top_book_indices = np.argsort(book_scores)[::-1][:5]
            
            for idx in top_book_indices:
                if book_scores[idx] > 0:
                    book = self.books_df.iloc[idx]
                    results.append(self.format_book_details(book) | {'relevance': book_scores[idx]})

        return results if results else "No matching results found."

    def parse_sort_criteria(self, query):
        """Extract sorting criteria from query."""
        sort_patterns = {
            'budget': r'(?:sort|order)?\s*by\s+budget\s*(?:desc|asc)?',
            'rating': r'(?:sort|order)?\s*by\s+rating\s*(?:desc|asc)?',
            'date': r'(?:sort|order)?\s*by\s+(?:date|release|publication)\s*(?:desc|asc)?',
            'relevance': r'(?:sort|order)?\s*by\s+relevance\s*(?:desc|asc)?'
        }
        
        for criteria, pattern in sort_patterns.items():
            match = re.search(pattern, query.lower())
            if match:
                desc = 'asc' not in match.group(0)
                return criteria, desc
        
        # Default sorting based on query type
        if 'budget' in query.lower():
            return 'budget', True
        elif 'rating' in query.lower():
            return 'rating', True
        elif any(word in query.lower() for word in ['date', 'release', 'published']):
            return 'date', True
        return 'relevance', True

    def find_exact_match(self, title, parsed):
        """Find exact matches for a title in either movies or books."""
        if parsed['type'] in [None, 'movie']:
            movie = self.movies_df[self.movies_df['title'].str.contains(title, case=False, na=False)]
            if not movie.empty:
                movie = movie.iloc[0]
                if parsed['attribute'] == 'rating':
                    return f"The rating of '{movie['title']}' is {movie['vote_average']:.1f}/10"
                elif parsed['attribute'] == 'year':
                    year = movie['release_date'].split('-')[0] if pd.notna(movie['release_date']) else 'unknown'
                    return f"'{movie['title']}' was released in {year}"
                elif parsed['attribute'] == 'overview':
                    return f"Here's what '{movie['title']}' is about:\n{movie['overview']}"
                elif parsed['attribute'] == 'budget':
                    if pd.notna(movie['budget']) and movie['budget'] > 0:
                        return f"The budget of '{movie['title']}' was ${movie['budget']:,.0f}"
                    return f"The budget of '{movie['title']}' is unknown"
                else:
                    return self.format_movie_details(movie)

        if parsed['type'] in [None, 'book']:
            book = self.books_df[self.books_df['title'].str.contains(title, case=False, na=False)]
            if not book.empty:
                book = book.iloc[0]
                if parsed['attribute'] == 'rating':
                    return f"The rating of '{book['title']}' is {book['average_rating']:.1f}/5"
                elif parsed['attribute'] == 'year':
                    return f"'{book['title']}' was published in {book['publication_date']}"
                else:
                    return self.format_book_details(book)
        
        return None

    def format_movie_details(self, movie):
        return {
            'type': 'movie',
            'title': movie['title'],
            'rating': f"{movie['vote_average']:.1f}/10" if pd.notna(movie['vote_average']) else 'N/A',
            'release_date': movie['release_date'],
            'runtime': f"{movie['runtime']} minutes" if pd.notna(movie['runtime']) else 'Unknown',
            'overview': movie['overview'],
            'budget': f"${movie['budget']:,.0f}" if pd.notna(movie['budget']) and movie['budget'] > 0 else 'Unknown'
        }

    def format_book_details(self, book):
        return {
            'type': 'book',
            'title': book['title'],
            'author': book['authors'],
            'rating': f"{book['average_rating']:.1f}/5" if pd.notna(book['average_rating']) else 'N/A',
            'publication_date': book['publication_date'],
            'pages': book['num_pages']
        }

def main():
    print("\n=== Welcome to the Books and Movies Search Engine ===")
    print("\nYou can ask questions like:")
    print("- What is the rating of [movie/book title]?")
    print("- What is the budget of [movie title]?")
    print("- When was [movie/book title] released/published?")
    print("- Movies similar to [movie title].")
    print("- Comedy hit and latest movies.")
    print("- Sci-fi movies with rating above 7.5.")
    print("\nVisualization commands:")
    print("- Show timeline for franchise [franchise name]")
    print("- Show timeline for author [author name]")
    print("- Show genre graph for [movies/books]\n")
    
    search_engine = SearchEngine()
    visualizer = TimelineVisualizer(search_engine.books_df, search_engine.movies_df)
    
    # Set matplotlib backend to a non-interactive backend if needed
    import matplotlib
    matplotlib.use('Agg')  # Add this line
    
    while True:
        query = input("Enter your question (or type 'exit' to quit): ")
        
        if query.lower() == 'exit':
            print("Exiting the search engine. Goodbye!")
            break
            
        # Check for visualization commands
        timeline_match = re.search(r'show timeline for (franchise|author) (.+)', query.lower())
        genre_graph_match = re.search(r'show genre graph for (movies|books)', query.lower())
        
        if timeline_match:
            vis_type, name = timeline_match.groups()
            if vis_type == 'franchise':
                result = visualizer.create_franchise_timeline(name)
            else:  # author
                result = visualizer.create_author_timeline(name)
                if isinstance(result, dict):
                    print(result['message'])
                    print("\nAuthor Statistics:")
                    for key, value in result['stats'].items():
                        print(f"{key}: {value}")
                    print("\nBooks in chronological order:")
                    for book in result['books']:
                        print(f"- {book['title']} ({book['parsed_date'].year}): {book['average_rating']:.1f}/5")
                else:
                    print(result)
        
        # Add this block to handle genre graph visualization
        elif genre_graph_match:
            media_type = genre_graph_match.group(1)
            try:
                result = visualizer.create_genre_graph(media_type)
                print(result)
            except Exception as e:
                print(f"Error creating genre graph: {str(e)}")
            
        else:
            # Regular search
            results = search_engine.search(query)
            if isinstance(results, str):
                print(results)
            elif isinstance(results, list):
                for result in results:
                    if isinstance(result, dict):
                        if result['type'] == 'movie':
                            print(f"\nMovie: {result['title']}")
                            print(f"Rating: {result['rating']}")
                            print(f"Release Date: {result['release_date']}")
                            if 'budget' in result:
                                print(f"Budget: {result['budget']}")
                            if 'overview' in result:
                                print(f"Overview: {result['overview']}")
                        elif result['type'] == 'book':
                            print(f"\nBook: {result['title']}")
                            print(f"Author: {result['author']}")
                            print(f"Rating: {result['rating']}")
                            print(f"Publication Date: {result['publication_date']}")
                        print("-" * 80)
                    else:
                        print(result)
            else:
                print(results)

if __name__ == "__main__":
    main()