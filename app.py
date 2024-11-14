from flask import Flask, render_template, request, jsonify
import pandas as pd
from se import SearchEngine, TimelineVisualizer, UserPreferences

app = Flask(__name__)

# Initialize the search engine
search_engine = SearchEngine()
visualizer = TimelineVisualizer(search_engine.books_df, search_engine.movies_df)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    if not query:
        return jsonify({'error': 'No query provided'})
    
    results = search_engine.search(query)
    
    # Convert results to JSON-serializable format
    if isinstance(results, list):
        formatted_results = []
        for result in results:
            if isinstance(result, dict):
                # Handle datetime objects and other non-serializable types
                formatted_result = {}
                for key, value in result.items():
                    if pd.isna(value):
                        formatted_result[key] = None
                    elif isinstance(value, pd.Timestamp):
                        formatted_result[key] = value.strftime('%Y-%m-%d')
                    else:
                        formatted_result[key] = value
                formatted_results.append(formatted_result)
            else:
                formatted_results.append(str(result))
        return jsonify({'results': formatted_results})
    else:
        return jsonify({'results': [str(results)]})

@app.route('/author-timeline', methods=['POST'])
def author_timeline():
    author = request.form.get('author')
    if not author:
        return jsonify({'error': 'No author provided'})
    
    result = visualizer.create_author_timeline(author)
    return jsonify(result)

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    username = request.form.get('username')
    if not username:
        return jsonify({'error': 'No username provided'})
    
    recommendations = search_engine.get_personalized_recommendations(username)
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)