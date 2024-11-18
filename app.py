import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import your existing classes
from se import SearchEngine, TimelineVisualizer

def init_session_state():
    """Initialize session state variables."""
    if 'search_engine' not in st.session_state:
        st.session_state.search_engine = SearchEngine()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = TimelineVisualizer(
            st.session_state.search_engine.books_df,
            st.session_state.search_engine.movies_df
        )

def show_search_results(results):
    """Display search results in a formatted way with improved styling."""
    st.markdown("""
    <style>
    .result-card {
        background-color: #1A1A1D;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
        color: #FFFFFF;
    }
    .result-card:hover {
        transform: scale(1.02);
    }
    .movie-icon { color: #FF6B6B; font-size: 24px; margin-right: 10px; }
    .book-icon { color: #4ECDC4; font-size: 24px; margin-right: 10px; }
    .result-title {
        font-size: 20px;
        font-weight: bold;
    }
    .result-details {
        margin-top: 10px;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

    if isinstance(results, str):
        st.warning(results)
    elif isinstance(results, list):
        for result in results:
            if isinstance(result, dict):
                title = result.get('title', 'Title Not Available')
                st.markdown(f"""
                <div class="result-card">
                    <span class="{'movie-icon' if result['type'] == 'movie' else 'book-icon'}">
                        {"🎬" if result['type'] == 'movie' else "📚"}
                    </span>
                    <span class="result-title">{title}</span>
                </div>
                """, unsafe_allow_html=True)
                
                if result['type'] == 'movie':
                    st.markdown(f"""
                    <div class="result-details">
                    <strong>Rating:</strong> {result.get('rating', 'N/A')} ⭐<br>
                    <strong>Release Date:</strong> {result.get('release_date', 'Unknown')}<br>
                    {"<strong>Budget:</strong> $" + str(result['budget']) if 'budget' in result else ""}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if 'overview' in result:
                        with st.expander("Show Overview"):
                            st.write(result['overview'])
                
                elif result['type'] == 'book':
                    st.markdown(f"""
                    <div class="result-details">
                    <strong>Author:</strong> {result.get('author', 'Unknown')}<br>
                    <strong>Rating:</strong> {result.get('rating', 'N/A')} ⭐<br>
                    <strong>Publication Date:</strong> {result.get('publication_date', 'Unknown')}
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.write(result)
    else:
        st.write(results)



def create_franchise_timeline_plotly(franchise_data):
    """Create an interactive timeline visualization using Plotly."""
    if not franchise_data or not isinstance(franchise_data, list):
        return None, "No data found for this franchise."
    
    fig = go.Figure()
    dates = []
    for item in franchise_data:
        try:
            date = pd.to_datetime(item['release_date'])
            dates.append(date)
        except:
            return None, "Invalid date format in franchise data"
    
    # Colorful marker visualization
    fig.add_trace(go.Scatter(
        x=dates,
        y=[float(item['rating']) for item in franchise_data],
        mode='markers+text',
        text=[item['title'] for item in franchise_data],
        textposition="top center",
        marker=dict(
            size=15, 
            color=[float(item['rating']) for item in franchise_data],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Rating')
        ),
        hovertemplate="<b>%{text}</b><br>Release Date: %{x|%Y-%m-%d}<br>Rating: %{y}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"{franchise_data[0].get('franchise', 'Unknown')} Franchise Timeline",
        xaxis_title="Release Date",
        yaxis_title="Rating",
        showlegend=False,
        hovermode='closest',
        height=600,
        template="plotly_white",
        xaxis=dict(type='date', tickformat='%Y-%m-%d')
    )
    
    stats = {
        'Total Movies': len(franchise_data),
        'Average Rating': sum(float(m['rating']) for m in franchise_data) / len(franchise_data),
        'Release Span': f"{min(dates).strftime('%Y-%m-%d')} - {max(dates).strftime('%Y-%m-%d')}",
        'Highest Rated': max(franchise_data, key=lambda x: float(x['rating']))['title'],
        'Most Recent': max(franchise_data, key=lambda x: pd.to_datetime(x['release_date']))['title'],
        'First Released': min(franchise_data, key=lambda x: pd.to_datetime(x['release_date']))['title']
    }
    
    return fig, {'movies': franchise_data, 'stats': stats}

def create_author_timeline_plotly(result):
    """Create an interactive author timeline visualization using Plotly."""
    if not isinstance(result, dict) or 'books' not in result:
        return None, result
    
    if not result['books']:
        return None, "No books found for this author."
    
    author_name = result['books'][0].get('author', 'Unknown Author') if result['books'] else 'Unknown Author'
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[book.get('parsed_date', datetime.now()) for book in result['books']],
        y=[book.get('average_rating', 0) for book in result['books']],
        mode='markers+text',
        text=[book.get('title', 'Unknown Title') for book in result['books']],
        textposition="top center",
        marker=dict(
            size=15, 
            color=[book.get('average_rating', 0) for book in result['books']],
            colorscale='Plasma',
            showscale=True,
            colorbar=dict(title='Rating')
        ),
        hovertemplate="<b>%{text}</b><br>Publication Date: %{x}<br>Rating: %{y}<extra></extra>"
    ))
    
    fig.update_layout(
        title=f"Timeline of {author_name}'s Books",
        xaxis_title="Publication Date",
        yaxis_title="Rating",
        showlegend=False,
        hovermode='closest',
        height=600,
        template="plotly_white"
    )
    
    return fig, result

def main():
    # Custom page config with a professional look
    st.set_page_config(
        page_title="Books & Movies Explorer",
        page_icon="🌟",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for a more professional look
    st.markdown("""
    <style>
    .reportview-container {
        background-color: #F0F2F6;
    }
    .sidebar .sidebar-content {
        background-color: #FFFFFF;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        border: 1px solid #ddd;
        padding: 10px;
    }
    h1, h2, h3 {
        color: #2C3E50;
    }
    </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    
    st.title("📚🎬 Books & Movies Explorer")
    st.markdown("*Discover, Search, and Explore Cinematic and Literary Worlds*")
    
    # Sidebar with improved navigation using selectbox
    page = st.sidebar.selectbox(
        "Navigate",
        ["Search", "Franchise Timeline", "Author Timeline"],
        index=0,
        format_func=lambda x: f"🔍 {x}" if x == "Search" else f"{'🎬' if x == 'Franchise Timeline' else '📚'} {x}"
    )
    
    if page == "Search":
        st.subheader("🔍 Intelligent Search")
        st.markdown("""
        ### Quick Search Tips:
        - Ask about movie or book ratings
        - Explore movie budgets
        - Find release or publication dates
        - Get movie recommendations
        """)
        
        query = st.text_input("Enter your search query:", key="search_query", 
                               placeholder="What would you like to know?")
        
        if st.button("Search", key="search_button") or query:
            with st.spinner("Searching our comprehensive database..."):
                results = st.session_state.search_engine.search(query)
                show_search_results(results)
    
    elif page == "Franchise Timeline":
        st.subheader("🎬 Franchise Journey")
        franchise = st.text_input("Enter franchise name:", key="franchise_input", 
                                  placeholder="e.g., Marvel, Star Wars")
        
        if st.button("Generate Timeline", key="franchise_timeline_button") or franchise:
            with st.spinner("Crafting franchise timeline..."):
                try:
                    franchise_data = st.session_state.visualizer.create_franchise_timeline(franchise)
                    
                    if isinstance(franchise_data, list):
                        fig, result = create_franchise_timeline_plotly(franchise_data)
                        if fig is not None:
                            stats = result.get('stats', {})
                            if stats:
                                st.subheader("Franchise Insights")
                                cols = st.columns(3)
                                metrics = [
                                    ("Total Movies", stats['Total Movies']),
                                    ("Avg Rating", f"{stats['Average Rating']:.2f}/10"),
                                    ("Release Span", stats['Release Span']),
                                    ("Highest Rated", stats['Highest Rated']),
                                    ("Most Recent", stats['Most Recent']),
                                    ("First Released", stats['First Released'])
                                ]
                                
                                for i, (label, value) in enumerate(metrics):
                                    cols[i % 3].metric(label, value)
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Could not generate timeline visualization.")
                    else:
                        st.error("No data available for this franchise.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    
    elif page == "Author Timeline":
        st.subheader("📚 Author's Literary Journey")
        author = st.text_input("Enter author name:", key="author_timeline_input", 
                                placeholder="e.g., Stephen King")
        
        if st.button("Generate Timeline", key="author_timeline_button") or author:
            with st.spinner("Mapping author's literary path..."):
                try:
                    result = st.session_state.visualizer.create_author_timeline(author)
                    
                    if isinstance(result, dict):
                        fig, result = create_author_timeline_plotly(result)
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Could not generate timeline visualization.")
                    else:
                        st.error("No data available for this author.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    # Professional footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("💡 Powered by Data Intelligence")
    st.sidebar.markdown("© 2024 Books & Movies Explorer")

if __name__ == "__main__":
    main()