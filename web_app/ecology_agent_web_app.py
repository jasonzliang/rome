import streamlit as st
import json
import os
from pathlib import Path
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict
import re
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import multiprocessing
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Ecology Agent Graph Explorer", layout="wide")

@st.cache_data
def load_graphs(directory, pattern='*.graph_iter*.json'):
    """Load all graph JSON files from directory."""
    graphs = {}
    iter_pattern = re.compile(r'graph_iter(\d+)\.json')

    for file in Path(directory).glob(pattern):
        match = iter_pattern.search(file.name)
        if match:
            iteration = int(match.group(1))
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    graphs[iteration] = json.load(f)
            except Exception as e:
                st.warning(f"Failed to load {file.name}: {e}")

    return dict(sorted(graphs.items()))

def create_network_graph(data, layout_type='spring'):
    """Create interactive network visualization."""
    G = nx.DiGraph()

    # Add nodes with attributes
    for node in data['nodes']:
        G.add_node(node['url'],
                   depth=node['depth'],
                   insights=node['insights'])

    # Add edges with validation
    for edge in data['edges']:
        if edge['source'] in G.nodes() and edge['target'] in G.nodes():
            G.add_edge(edge['source'], edge['target'],
                       reason=edge['reason'])

    # Layout
    layouts = {
        'spring': nx.spring_layout(G, k=2, iterations=50),
        'circular': nx.circular_layout(G),
        'kamada': nx.kamada_kawai_layout(G),
        'shell': nx.shell_layout(G)
    }
    pos = layouts.get(layout_type, nx.spring_layout(G))

    # Create edges trace
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create nodes trace
    node_x, node_y, node_text, node_color = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        title = node.split('/')[-1].replace('_', ' ')
        depth = G.nodes[node]['depth']
        node_text.append(f"{title}<br>Depth: {depth}")
        node_color.append(depth)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=20,
            color=node_color,
            colorbar=dict(
                thickness=15,
                title=dict(text='Depth', side='right'),
                xanchor='left'
            ),
            line=dict(width=2, color='white')))

    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0, l=0, r=0, t=0),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=600))

    return fig, G

def analyze_evolution(graphs):
    """Analyze how graphs evolve over iterations."""
    metrics = {
        'iteration': [],
        'num_nodes': [],
        'num_edges': [],
        'avg_depth': [],
        'max_depth': [],
        'density': []
    }

    for iter_num, data in graphs.items():
        metrics['iteration'].append(iter_num)
        metrics['num_nodes'].append(len(data['nodes']))
        metrics['num_edges'].append(len(data['edges']))

        depths = [n['depth'] for n in data['nodes']]
        metrics['avg_depth'].append(np.mean(depths) if depths else 0)
        metrics['max_depth'].append(max(depths) if depths else 0)

        n = len(data['nodes'])
        e = len(data['edges'])
        metrics['density'].append(e / (n * (n - 1)) if n > 1 else 0)

    return pd.DataFrame(metrics)

def create_evolution_plots(df):
    """Create plots showing graph evolution."""
    fig = go.Figure()

    # Add nodes trace
    fig.add_trace(go.Scatter(
        x=df['iteration'],
        y=df['num_nodes'],
        name='Nodes',
        mode='lines+markers',
        line=dict(width=3, color='#1f77b4'),
        marker=dict(size=10, color='#1f77b4'),
        hovertemplate='<b>Iteration %{x}</b><br>Nodes: %{y}<extra></extra>'))

    # Add edges trace
    fig.add_trace(go.Scatter(
        x=df['iteration'],
        y=df['num_edges'],
        name='Edges',
        mode='lines+markers',
        line=dict(width=3, color='#ff7f0e'),
        marker=dict(size=10, color='#ff7f0e'),
        hovertemplate='<b>Iteration %{x}</b><br>Edges: %{y}<extra></extra>'))

    fig.update_layout(
        title='Graph Growth Over Time',
        xaxis_title='Iteration',
        yaxis_title='Count',
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ))

    return fig

def create_depth_plot(df):
    """Create depth evolution plot."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['iteration'], y=df['avg_depth'],
        name='Average Depth', mode='lines+markers',
        line=dict(width=3),
        marker=dict(size=8)))

    fig.add_trace(go.Scatter(
        x=df['iteration'], y=df['max_depth'],
        name='Max Depth', mode='lines+markers',
        line=dict(width=3),
        marker=dict(size=8)))

    fig.update_layout(
        title='Exploration Depth Over Time',
        xaxis_title='Iteration',
        yaxis_title='Depth (steps from origin)',
        hovermode='x unified',
        height=400,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ))

    return fig

@st.cache_data
def train_word2vec_model(graphs):
    """Train Word2Vec on insights and topics."""
    sentences = []

    for data in graphs.values():
        for node in data['nodes']:
            # Tokenize insights
            insights_text = node['insights'].lower()
            # Simple tokenization - filter out very common words
            words = re.findall(r'\b[a-z]{3,}\b', insights_text)
            if len(words) > 5:  # Only include meaningful sentences
                sentences.append(words)

            # Add URL topic as context
            topic = node['url'].split('/')[-1].replace('_', ' ').lower()
            topic_words = topic.split()
            if topic_words:
                sentences.append(topic_words)

    # Check if we have enough data
    if len(sentences) < 2:
        return None

    # Train Word2Vec with error handling
    try:
        model = Word2Vec(
            sentences=sentences,
            vector_size=100,
            window=5,
            min_count=2,
            workers=multiprocessing.cpu_count(),
            epochs=20,
            seed=42
        )
        return model
    except Exception as e:
        st.error(f"Error training Word2Vec model: {e}")
        return None

@st.cache_data
def analyze_topic_relationships(graphs, _model):
    """Analyze topic relationships using Word2Vec."""
    if _model is None:
        return {}, {}

    # Get all unique topics
    all_topics = set()
    topic_to_url = {}

    for data in graphs.values():
        for node in data['nodes']:
            topic = node['url'].split('/')[-1].replace('_', ' ')
            all_topics.add(topic)
            topic_to_url[topic] = node['url']

    # Get embeddings for topics
    topic_vectors = {}
    for topic in all_topics:
        words = topic.lower().split()
        # Average word vectors for multi-word topics
        vectors = []
        for word in words:
            try:
                if word in _model.wv:
                    vectors.append(_model.wv[word])
            except:
                continue

        if vectors:
            topic_vectors[topic] = np.mean(vectors, axis=0)

    return topic_vectors, topic_to_url

def create_topic_similarity_network(topic_vectors, topic_to_url, top_n=50):
    """Create network graph of similar topics."""
    topics = list(topic_vectors.keys())[:top_n]  # Limit for visualization

    if len(topics) < 2:
        return None

    # Calculate similarity matrix
    vectors = [topic_vectors[t] for t in topics]
    sim_matrix = cosine_similarity(vectors)

    # Create graph with top similarities
    G = nx.Graph()
    for i, topic in enumerate(topics):
        G.add_node(topic, url=topic_to_url.get(topic, ''))

    # Add edges for high similarity
    threshold = 0.7
    for i in range(len(topics)):
        for j in range(i+1, len(topics)):
            if sim_matrix[i][j] > threshold:
                G.add_edge(topics[i], topics[j],
                          weight=float(sim_matrix[i][j]))

    return G, sim_matrix, topics

def create_topic_embedding_plot(topic_vectors):
    """Create 2D visualization of topic embeddings using t-SNE."""
    if len(topic_vectors) < 2:
        return None

    topics = list(topic_vectors.keys())
    vectors = np.array([topic_vectors[t] for t in topics])

    # Use t-SNE for dimensionality reduction
    try:
        if len(vectors) > 2:
            perplexity = min(30, max(5, len(vectors) - 1))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            coords = tsne.fit_transform(vectors)
        else:
            coords = vectors[:, :2]  # Just use first 2 dimensions
    except Exception as e:
        st.warning(f"t-SNE failed: {e}. Using first 2 dimensions instead.")
        coords = vectors[:, :2]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode='markers',
        hoverinfo='text',
        hovertext=[f"<b>{topic}</b><br>Index: {i}" for i, topic in enumerate(topics)],
        marker=dict(
            size=10,
            color=list(range(len(topics))),  # Convert range to list
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=dict(text="Topic Index"))
        )
    ))

    fig.update_layout(
        title='Topic Embedding Space (t-SNE)',
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        height=600,
        hovermode='closest'
    )

    return fig

# Main app
st.title("🌿 Ecology Agent Graph Explorer")

# Sidebar
st.sidebar.header("Configuration")

# Directory input with file browser option
st.sidebar.subheader("📁 Select Directory")

# Initialize session state
home_dir = os.path.expanduser("~")
if 'current_dir' not in st.session_state:
    st.session_state['current_dir'] = os.path.join(home_dir, "Desktop")
if 'selected_dir' not in st.session_state:
    st.session_state['selected_dir'] = None

# Option 1: Manual path input (for local runs)
st.sidebar.markdown("**Option 1: Enter Directory Path**")
manual_path = st.sidebar.text_input(
    "Directory path:",
    placeholder="/path/to/your/graph/directory",
    help="Paste the full path to your graph directory")

if manual_path and st.sidebar.button("📂 Load from Path", use_container_width=True):
    manual_path = os.path.expanduser(manual_path)
    if os.path.exists(manual_path) and os.path.isdir(manual_path):
        st.session_state['selected_dir'] = manual_path
        st.rerun()
    else:
        st.sidebar.error("❌ Invalid directory path")

st.sidebar.markdown("---")

# Option 2: Directory browser
st.sidebar.markdown("**Option 2: Browse Directories**")

# Directory browser
current_dir = st.session_state.get('current_dir', home_dir)
current_dir = os.path.expanduser(current_dir)

# Show current directory
st.sidebar.text_area(
    "Currently browsing:",
    value=current_dir,
    height=60,
    disabled=True,
    help="Currently browsing this directory")

if os.path.exists(current_dir) and os.path.isdir(current_dir):
    # List directories
    try:
        items = os.listdir(current_dir)
        dirs = sorted([d for d in items
                      if os.path.isdir(os.path.join(current_dir, d))
                      and not d.startswith('.')])
        dirs.insert(0, "..")  # Add parent directory option
        dirs.insert(0, ".")   # Add current directory option

        selected_dir = st.sidebar.selectbox(
            "📂 Navigate to:",
            dirs,
            help="Select '..' to go up, '.' to use current directory, or choose a subdirectory")

        # Navigation buttons
        col1, col2, col3 = st.sidebar.columns(3)

        if selected_dir == "..":
            if col1.button("⬆️ Up", use_container_width=True):
                st.session_state['current_dir'] = os.path.dirname(current_dir)
                st.rerun()
        elif selected_dir == ".":
            if col2.button("✅ Use", use_container_width=True):
                st.session_state['selected_dir'] = current_dir
                st.rerun()
        else:
            if col1.button("📂 Open", use_container_width=True):
                st.session_state['current_dir'] = os.path.join(current_dir, selected_dir)
                st.rerun()

            if col2.button("✅ Use", use_container_width=True):
                st.session_state['selected_dir'] = os.path.join(current_dir, selected_dir)
                st.rerun()

        # Quick jump to common locations
        st.sidebar.markdown("**Quick Jump:**")
        jump_cols = st.sidebar.columns(2)

        if jump_cols[0].button("🏠 Home", use_container_width=True):
            st.session_state['current_dir'] = home_dir
            st.rerun()

        if jump_cols[1].button("🖥️ Desktop", use_container_width=True):
            st.session_state['current_dir'] = os.path.join(home_dir, "Desktop")
            st.rerun()

    except PermissionError:
        st.sidebar.error("❌ Permission denied")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")
else:
    st.sidebar.error("❌ Directory does not exist")
    if st.sidebar.button("Reset to Home"):
        st.session_state['current_dir'] = home_dir
        st.rerun()

# Use selected directory or current
directory = st.session_state.get('selected_dir')
if directory is None:
    directory = current_dir

# Validate directory
if not directory or not os.path.exists(directory):
    st.error(f"❌ Please select a directory using the browser on the left")
    st.info("💡 Navigate to your graph directory and click '✅ Use Current' or '✅ Use This'")
    st.stop()

# Show current directory being used
st.sidebar.markdown("---")
st.sidebar.markdown("**Current Directory:**")
st.sidebar.code(directory, language=None)

# Advanced options
with st.sidebar.expander("⚙️ Advanced Options"):
    file_pattern = st.text_input(
        "File Pattern",
        value="*.graph_iter*.json",
        help="Glob pattern to match JSON files")

    if st.button("🔄 Reload Files"):
        st.cache_data.clear()
        st.rerun()

# Load graphs
with st.spinner("Loading graphs..."):
    graphs = load_graphs(directory, file_pattern)

if not graphs:
    st.error("❌ No graph files found in directory")
    st.info(f"Looking for files matching pattern: `{file_pattern}`")

    # Show what files are in the directory
    try:
        all_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        if all_files:
            st.write("**JSON files found in directory:**")
            for f in sorted(all_files)[:20]:  # Show first 20
                st.write(f"- `{f}`")
            if len(all_files) > 20:
                st.write(f"... and {len(all_files) - 20} more")
        else:
            st.write("No JSON files found in this directory")
    except Exception as e:
        st.error(f"Error reading directory: {e}")
    st.stop()

st.sidebar.success(f"✅ Loaded {len(graphs)} iterations")
st.sidebar.markdown(f"**Iterations:** {min(graphs.keys())} - {max(graphs.keys())}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Single Graph View", "📈 Evolution Analysis", "🔎 Detailed Insights", "🧠 Topic Relationships"])

with tab1:
    st.header("Individual Graph Visualization")

    col1, col2 = st.columns([2, 1])

    with col1:
        iteration = st.select_slider(
            "Select Iteration",
            options=list(graphs.keys()),
            value=max(graphs.keys()))

    with col2:
        layout = st.selectbox(
            "Layout Algorithm",
            ['kamada', 'spring', 'circular', 'shell'],
            index=0)

    data = graphs[iteration]

    # Display graph
    fig, G = create_network_graph(data, layout)
    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes", len(data['nodes']))
    col2.metric("Edges", len(data['edges']))
    col3.metric("Max Depth", max([n['depth'] for n in data['nodes']]))
    col4.metric("Avg Degree", f"{2*len(data['edges'])/len(data['nodes']):.2f}")

    # Node details
    st.subheader("Node Details")

    # Create more readable node labels
    node_labels = {n['url']: n['url'].split('/')[-1].replace('_', ' ')
                   for n in data['nodes']}

    selected_label = st.selectbox(
        "Select Node",
        list(node_labels.values()))

    # Get URL from label
    selected_node = [url for url, label in node_labels.items()
                     if label == selected_label][0]

    node_data = next(n for n in data['nodes'] if n['url'] == selected_node)

    st.write(f"**Title:** {selected_label}")
    st.write(f"**URL:** [{node_data['url']}]({node_data['url']})")
    st.write(f"**Depth:** {node_data['depth']}")
    st.write("**Insights:**")
    with st.expander("View full insights", expanded=True):
        st.markdown(node_data['insights'])

    # Connected edges
    connected_edges = [e for e in data['edges']
                      if e['source'] == selected_node or e['target'] == selected_node]

    if connected_edges:
        st.subheader("Connected Edges")
        for i, edge in enumerate(connected_edges, 1):
            direction = "→" if edge['source'] == selected_node else "←"
            other = edge['target'] if edge['source'] == selected_node else edge['source']
            other_label = other.split('/')[-1].replace('_', ' ')

            with st.expander(f"{i}. {direction} {other_label}"):
                st.write(f"**Connection:** {edge['source'].split('/')[-1]} → {edge['target'].split('/')[-1]}")
                st.write(f"**Reason:** {edge['reason']}")

with tab2:
    st.header("Evolution Analysis")

    st.info("""
    📊 **Understanding the Metrics:**
    - **Nodes**: Number of unique topics/Wikipedia pages discovered
    - **Edges**: Number of connections/links between topics
    - **Average Depth**: Mean distance of all nodes from the starting point (how spread out the exploration is)
    - **Max Depth**: Furthest distance reached from the starting point (how deep the exploration went)
    """)

    # Compute metrics
    df_metrics = analyze_evolution(graphs)

    # Growth plots
    st.plotly_chart(create_evolution_plots(df_metrics), use_container_width=True)

    # Depth plots
    st.plotly_chart(create_depth_plot(df_metrics), use_container_width=True)

    # Density plot
    fig_density = go.Figure()
    fig_density.add_trace(go.Scatter(
        x=df_metrics['iteration'],
        y=df_metrics['density'],
        mode='lines+markers',
        name='Graph Density',
        line=dict(width=3),
        marker=dict(size=10)))

    fig_density.update_layout(
        title='Graph Density Over Time',
        xaxis_title='Iteration',
        yaxis_title='Density',
        height=400)

    st.plotly_chart(fig_density, use_container_width=True)

with tab3:
    st.header("Detailed Insights Explorer")

    # Iteration comparison
    st.subheader("Compare Iterations")
    col1, col2 = st.columns(2)

    with col1:
        iter1 = st.selectbox("First Iteration",
                            list(graphs.keys()),
                            key='iter1')
    with col2:
        iter2 = st.selectbox("Second Iteration",
                            list(graphs.keys()),
                            index=min(len(graphs)-1, 1),
                            key='iter2')

    if iter1 != iter2:
        data1 = graphs[iter1]
        data2 = graphs[iter2]

        urls1 = set(n['url'] for n in data1['nodes'])
        urls2 = set(n['url'] for n in data2['nodes'])

        new_nodes = urls2 - urls1
        removed_nodes = urls1 - urls2
        common_nodes = urls1 & urls2

        col1, col2, col3 = st.columns(3)
        col1.metric("New Nodes", len(new_nodes))
        col2.metric("Removed Nodes", len(removed_nodes))
        col3.metric("Common Nodes", len(common_nodes))

        if new_nodes:
            with st.expander(f"New nodes in iteration {iter2}"):
                for url in sorted(new_nodes):
                    st.write(f"- {url.split('/')[-1].replace('_', ' ')}")

    st.markdown("---")

    # Topic tracking
    st.subheader("Topic Evolution")

    # Extract unique URLs across all iterations
    all_urls = set()
    for data in graphs.values():
        all_urls.update([n['url'] for n in data['nodes']])

    # Track when each URL appears
    url_timeline = defaultdict(list)
    for iter_num, data in graphs.items():
        for node in data['nodes']:
            url_timeline[node['url']].append(iter_num)

    # Show topics
    topic_data = []
    for url, iterations in url_timeline.items():
        topic_data.append({
            'Topic': url.split('/')[-1].replace('_', ' '),
            'First Seen': min(iterations),
            'Last Seen': max(iterations),
            'Appearances': len(set(iterations))
        })

    df_topics = pd.DataFrame(topic_data).sort_values('First Seen')
    st.dataframe(df_topics, use_container_width=True)

    # Cumulative topics
    fig_cumulative = go.Figure()
    cumulative_topics = []
    seen_urls = set()

    for iter_num in sorted(graphs.keys()):
        for node in graphs[iter_num]['nodes']:
            seen_urls.add(node['url'])
        cumulative_topics.append(len(seen_urls))

    fig_cumulative.add_trace(go.Scatter(
        x=sorted(graphs.keys()),
        y=cumulative_topics,
        mode='lines+markers',
        name='Cumulative Unique Topics'))

    fig_cumulative.update_layout(
        title='Cumulative Unique Topics Explored',
        xaxis_title='Iteration',
        yaxis_title='Number of Unique Topics',
        height=400)

    st.plotly_chart(fig_cumulative, use_container_width=True)

with tab4:
    st.header("Topic Relationships (Word2Vec Analysis)")

    st.info("🧠 Training Word2Vec model on insights and topics to discover semantic relationships...")

    # Train Word2Vec model
    with st.spinner("Training Word2Vec model..."):
        w2v_model = train_word2vec_model(graphs)

    if w2v_model is None:
        st.error("❌ Could not train Word2Vec model. Not enough data or training failed.")
        st.info("💡 Make sure your graphs contain sufficient text in the insights field.")
    else:
        topic_vectors, topic_to_url = analyze_topic_relationships(graphs, w2v_model)

        if len(topic_vectors) == 0:
            st.warning("⚠️ No topics could be embedded. Words from topics not found in vocabulary.")
            st.info(f"Vocabulary size: {len(w2v_model.wv)} words")
        else:
            st.success(f"✅ Model trained on {len(topic_vectors)} topics with {len(w2v_model.wv)} vocabulary terms")

            # Topic embedding visualization
            st.subheader("Topic Embedding Space")
            st.markdown("""
            **What you're seeing:**
            - Each point represents a topic (Wikipedia page) from your exploration
            - **Topic Index**: Simply a sequential number assigned to each topic (0, 1, 2, 3...)
            - **Position**: Topics that are semantically similar are placed closer together
            - **t-SNE**: Reduces 100-dimensional word embeddings to 2D for visualization

            💡 **How to interpret:** Topics close together have similar content/concepts, while distant topics are semantically different.
            """)

            embedding_fig = create_topic_embedding_plot(topic_vectors)
            if embedding_fig:
                st.plotly_chart(embedding_fig, use_container_width=True)

                # Show top topic keywords
                st.subheader("📚 Top Topic Keywords by Frequency")
                st.caption("Most frequently occurring words across all topics")

                # Extract all topic words
                topic_word_freq = {}
                for topic in topic_vectors.keys():
                    words = topic.lower().split()
                    for word in words:
                        if len(word) > 3:  # Filter short words
                            topic_word_freq[word] = topic_word_freq.get(word, 0) + 1

                # Sort and display top keywords
                if topic_word_freq:
                    sorted_keywords = sorted(topic_word_freq.items(), key=lambda x: x[1], reverse=True)[:20]

                    col1, col2 = st.columns(2)

                    # Create bar chart
                    with col1:
                        keywords_df = pd.DataFrame(sorted_keywords, columns=['Word', 'Frequency'])
                        fig_keywords = go.Figure()
                        fig_keywords.add_trace(go.Bar(
                            x=keywords_df['Frequency'],
                            y=keywords_df['Word'],
                            orientation='h',
                            marker=dict(
                                color=keywords_df['Frequency'],
                                colorscale='Viridis',
                                showscale=False
                            )
                        ))
                        fig_keywords.update_layout(
                            title='Top 20 Topic Keywords',
                            xaxis_title='Frequency',
                            yaxis_title='Keyword',
                            height=500,
                            yaxis={'categoryorder': 'total ascending'}
                        )
                        st.plotly_chart(fig_keywords, use_container_width=True)

                    # Show topics containing top keywords
                    with col2:
                        top_keyword = sorted_keywords[0][0]
                        st.markdown(f"**Topics containing '{top_keyword}':**")
                        matching_topics = [t for t in topic_vectors.keys() if top_keyword in t.lower()]
                        for topic in matching_topics[:10]:
                            st.write(f"• {topic}")
                        if len(matching_topics) > 10:
                            st.caption(f"... and {len(matching_topics) - 10} more")
            else:
                st.warning("Not enough topics for embedding visualization")

            # Topic similarity analysis
            st.subheader("Topic Similarity Analysis")

            if len(topic_vectors) > 1:
                col1, col2 = st.columns([1, 1])

                with col1:
                    # Select a topic to find similar topics
                    topics_list = sorted(topic_vectors.keys())
                    selected_topic = st.selectbox(
                        "Select a topic to find similar topics:",
                        topics_list)

                with col2:
                    top_k = st.slider("Number of similar topics", 5, min(20, len(topic_vectors)-1), min(10, len(topic_vectors)-1))

                if selected_topic and selected_topic in topic_vectors:
                    selected_vec = topic_vectors[selected_topic]

                    # Calculate similarities
                    similarities = []
                    for topic, vec in topic_vectors.items():
                        if topic != selected_topic:
                            try:
                                sim = cosine_similarity([selected_vec], [vec])[0][0]
                                similarities.append((topic, float(sim)))
                            except:
                                continue

                    if similarities:
                        # Sort by similarity
                        similarities.sort(key=lambda x: x[1], reverse=True)

                        st.markdown(f"**Most similar topics to '{selected_topic}':**")

                        # Create a bar chart
                        top_similar = similarities[:top_k]
                        similar_topics = [t[0] for t in top_similar]
                        similar_scores = [t[1] for t in top_similar]

                        fig_sim = go.Figure()
                        fig_sim.add_trace(go.Bar(
                            x=similar_scores,
                            y=similar_topics,
                            orientation='h',
                            marker=dict(
                                color=similar_scores,
                                colorscale='Viridis',
                                showscale=True
                            )
                        ))

                        fig_sim.update_layout(
                            title=f'Top {len(top_similar)} Most Similar Topics',
                            xaxis_title='Cosine Similarity',
                            yaxis_title='Topic',
                            height=400,
                            yaxis={'categoryorder': 'total ascending'}
                        )

                        st.plotly_chart(fig_sim, use_container_width=True)

                        # Show details
                        with st.expander("View similarity details"):
                            df_sim = pd.DataFrame(similarities[:top_k], columns=['Topic', 'Similarity'])
                            df_sim['URL'] = df_sim['Topic'].map(topic_to_url)
                            st.dataframe(df_sim, use_container_width=True)
                    else:
                        st.warning("Could not calculate similarities")
            else:
                st.warning("Need at least 2 topics for similarity analysis")

            # Similarity network
            st.subheader("Topic Similarity Network")
            st.markdown("Shows topics connected when their semantic similarity exceeds 0.7")

            if len(topic_vectors) > 1:
                max_topics = st.slider("Max topics to display", 10, min(100, len(topic_vectors)), min(30, len(topic_vectors)))

                result = create_topic_similarity_network(topic_vectors, topic_to_url, top_n=max_topics)

                if result:
                    G_sim, sim_matrix, topics = result

                    if len(G_sim.edges()) > 0:
                        # Layout
                        pos = nx.spring_layout(G_sim, k=2, iterations=50)

                        # Create edges
                        edge_x, edge_y = [], []
                        for edge in G_sim.edges():
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])

                        edge_trace = go.Scatter(
                            x=edge_x, y=edge_y,
                            line=dict(width=1, color='#888'),
                            hoverinfo='none',
                            mode='lines')

                        # Create nodes
                        node_x, node_y, node_text = [], [], []
                        for node in G_sim.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                            node_text.append(node)

                        node_trace = go.Scatter(
                            x=node_x, y=node_y,
                            mode='markers',
                            hoverinfo='text',
                            hovertext=node_text,
                            marker=dict(
                                size=15,
                                color='lightblue',
                                line=dict(width=2, color='white')))

                        fig_network = go.Figure(data=[edge_trace, node_trace],
                                               layout=go.Layout(
                                                   showlegend=False,
                                                   hovermode='closest',
                                                   margin=dict(b=0, l=0, r=0, t=0),
                                                   xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                   yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                   height=600))

                        st.plotly_chart(fig_network, use_container_width=True)

                        st.info(f"Network has {len(G_sim.nodes())} nodes and {len(G_sim.edges())} edges")
                    else:
                        st.warning("No topic pairs have similarity > 0.7. Try lowering the threshold or adding more topics.")
                else:
                    st.warning("Not enough topics for network visualization")

            # Word similarity search
            st.subheader("Word Similarity Search")
            st.markdown("Explore semantic relationships between individual words in the corpus")

            search_word = st.text_input("Enter a word to find similar terms:", "knowledge")

            if search_word and search_word.lower() in w2v_model.wv:
                try:
                    similar_words = w2v_model.wv.most_similar(search_word.lower(), topn=15)

                    df_words = pd.DataFrame(similar_words, columns=['Word', 'Similarity'])

                    fig_words = go.Figure()
                    fig_words.add_trace(go.Bar(
                        x=df_words['Similarity'],
                        y=df_words['Word'],
                        orientation='h',
                        marker=dict(color='skyblue')
                    ))

                    fig_words.update_layout(
                        title=f"Words most similar to '{search_word}'",
                        xaxis_title='Cosine Similarity',
                        yaxis_title='Word',
                        height=400,
                        yaxis={'categoryorder': 'total ascending'}
                    )

                    st.plotly_chart(fig_words, use_container_width=True)
                except Exception as e:
                    st.error(f"Error finding similar words: {e}")
            elif search_word:
                st.warning(f"Word '{search_word}' not found in vocabulary. Try another word.")

                # Show some example words
                try:
                    vocab_sample = list(w2v_model.wv.index_to_key[:20])
                    st.info(f"Example words in vocabulary: {', '.join(vocab_sample)}")
                except:
                    pass

st.sidebar.markdown("---")
st.sidebar.info("📊 Navigate through tabs to explore graphs, evolution, insights, and topic relationships using Word2Vec.")