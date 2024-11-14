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

        # Regular search
        parsed = self.parse_query(query)
        
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

    def parse_query(self, query):
        query = query.lower()

        parsed = {
            'type': None,
            'attribute': None,
            'title': None,
            'similar': False
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

        return parsed

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
    print("- Comedy movies with rating above X.\n")
    
    search_engine = SearchEngine()
    
    while True:
        query = input("Enter your question (or type 'exit' to quit): ")
        
        if query.lower() == 'exit':
            print("Exiting the search engine. Goodbye!")
            break
        
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
                    