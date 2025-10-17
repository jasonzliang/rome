from collections import defaultdict
import json
import multiprocessing
import os
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Ecology Agent Graph Explorer", layout="wide")

# ============================================================================
# DATA LOADING & PROCESSING
# ============================================================================

@st.cache_data
def load_graphs(directory, pattern='*.graph_iter*.json'):
    """Load all graph JSON files from directory."""
    graphs = {}
    iter_pattern = re.compile(r'graph_iter(\d+)\.json')

    for file in Path(directory).glob(pattern):
        if match := iter_pattern.search(file.name):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    graphs[int(match.group(1))] = json.load(f)
            except Exception as e:
                st.warning(f"Failed to load {file.name}: {e}")

    return dict(sorted(graphs.items()))

def create_graph_object(data):
    """Create NetworkX graph from data."""
    G = nx.DiGraph()
    G.graph['starting_url'] = data.get('starting_url')
    G.graph['iteration'] = data.get('iteration')

    for node in data['nodes']:
        G.add_node(node['url'], depth=node['depth'], insights=node['insights'])
    for edge in data['edges']:
        if edge['source'] in G.nodes() and edge['target'] in G.nodes():
            G.add_edge(edge['source'], edge['target'], reason=edge['reason'])
    return G

# ============================================================================
# VISUALIZATION COMPONENTS
# ============================================================================

def get_layout_position(G, layout_type='spring'):
    """Get node positions for different layouts."""
    layouts = {
        'spring': lambda: nx.spring_layout(G, k=2, iterations=50),
        'circular': lambda: nx.circular_layout(G),
        'kamada': lambda: nx.kamada_kawai_layout(G),
        'shell': lambda: nx.shell_layout(G)
    }
    return layouts.get(layout_type, layouts['spring'])()

def create_plotly_traces(G, pos):
    """Create Plotly edge and node traces."""
    # Edge trace
    edge_x, edge_y = [], []
    for src, tgt in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.8, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Node trace
    node_x, node_y, node_text, node_color = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        title = node.split('/')[-1].replace('_', ' ')
        depth = G.nodes[node]['depth']
        node_text.append(f"{title}<br>Depth: {depth}")
        node_color.append('red' if node == G.graph.get('starting_url') else depth)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=10,
            color=node_color,
            colorbar=dict(thickness=15, title=dict(text='Depth', side='right'), xanchor='left'),
            line=dict(width=2, color='white')))

    return edge_trace, node_trace

def create_network_graph(data, layout_type='spring'):
    """Create interactive network visualization."""
    G = create_graph_object(data)
    pos = get_layout_position(G, layout_type)
    edge_trace, node_trace = create_plotly_traces(G, pos)

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600))

    return fig, G

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def compute_graph_metrics(graphs):
    """Compute evolution metrics for all graphs."""
    metrics = defaultdict(list)

    for iter_num, data in graphs.items():
        depths = [n['depth'] for n in data['nodes']]
        n_nodes = len(data['nodes'])
        n_edges = len(data['edges'])

        metrics['iteration'].append(iter_num)
        metrics['num_nodes'].append(n_nodes)
        metrics['num_edges'].append(n_edges)
        metrics['avg_depth'].append(np.mean(depths) if depths else 0)
        metrics['max_depth'].append(max(depths) if depths else 0)
        metrics['density'].append(n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0)

    return pd.DataFrame(metrics)

def create_metric_plot(df, x_col, y_cols, title, y_label):
    """Generic function to create metric plots."""
    fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (col, name) in enumerate(y_cols):
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[col],
            name=name,
            mode='lines+markers',
            line=dict(width=3, color=colors[i % len(colors)]),
            marker=dict(size=10 if len(y_cols) > 1 else 8),
            hovertemplate=f'<b>Iteration %{{x}}</b><br>{name}: %{{y}}<extra></extra>'))

    fig.update_layout(
        title=title,
        xaxis_title='Iteration',
        yaxis_title=y_label,
        hovermode='x unified',
        height=500 if len(y_cols) == 2 else 400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                   bgcolor='rgba(255,255,255,0.8)' if len(y_cols) > 1 else None))

    return fig

# ============================================================================
# WORD2VEC & TOPIC ANALYSIS
# ============================================================================

@st.cache_data
def train_word2vec_model(graphs):
    """Train Word2Vec on insights and topics."""
    sentences = []

    for data in graphs.values():
        for node in data['nodes']:
            words = re.findall(r'\b[a-z]{3,}\b', node['insights'].lower())
            if len(words) > 5:
                sentences.append(words)

            topic_words = node['url'].split('/')[-1].replace('_', ' ').lower().split()
            if topic_words:
                sentences.append(topic_words)

    if len(sentences) < 2:
        return None

    try:
        return Word2Vec(
            sentences=sentences,
            vector_size=100,
            window=5,
            min_count=3,  # Reduce noise
            workers=multiprocessing.cpu_count(),
            epochs=15,  # More efficient
            sg=0,  # Skip-gram
            seed=42)
    except Exception as e:
        st.error(f"Error training Word2Vec model: {e}")
        return None

def get_topic_embeddings(graphs, model):
    """Extract topic vectors using Word2Vec."""
    if model is None:
        return {}, {}

    topic_vectors, topic_to_url = {}, {}

    for data in graphs.values():
        for node in data['nodes']:
            topic = node['url'].split('/')[-1].replace('_', ' ')
            topic_to_url[topic] = node['url']

            vectors = [model.wv[word] for word in topic.lower().split()
                      if word in model.wv]
            if vectors:
                topic_vectors[topic] = np.mean(vectors, axis=0)

    return topic_vectors, topic_to_url

def reduce_dimensions(vectors, method='tsne'):
    """Reduce dimensionality for visualization."""
    if len(vectors) <= 2:
        return vectors[:, :2]

    try:
        perplexity = min(30, max(5, len(vectors) - 1))
        return TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(vectors)
    except Exception as e:
        st.warning(f"t-SNE failed: {e}. Using first 2 dimensions.")
        return vectors[:, :2]

def create_similarity_network(topic_vectors, topic_to_url, top_n=50, threshold=0.7):
    """Create topic similarity network."""
    topics = list(topic_vectors.keys())[:top_n]
    if len(topics) < 2:
        return None

    vectors = [topic_vectors[t] for t in topics]
    sim_matrix = cosine_similarity(vectors)

    G = nx.Graph()
    for i, topic in enumerate(topics):
        G.add_node(topic, url=topic_to_url.get(topic, ''))
        for j in range(i+1, len(topics)):
            if sim_matrix[i][j] > threshold:
                G.add_edge(topics[i], topics[j], weight=float(sim_matrix[i][j]))

    return G, sim_matrix, topics

# ============================================================================
# UI COMPONENTS
# ============================================================================

def setup_directory_browser():
    """Setup directory browsing UI."""
    home_dir = os.path.expanduser("~")
    if 'current_dir' not in st.session_state:
        st.session_state['current_dir'] = os.path.join(home_dir, "Desktop")
    if 'selected_dir' not in st.session_state:
        st.session_state['selected_dir'] = None

    st.sidebar.subheader("📁 Select Directory")

    st.sidebar.markdown("**Option 1: Enter Directory Path**")
    manual_path = st.sidebar.text_input(
        "Directory path:",
        placeholder="/path/to/your/graph/directory",
        help="Paste the full path to your graph directory")

    if manual_path and st.sidebar.button("📂 Load from Path", width="stretch"):
        manual_path = os.path.expanduser(manual_path)
        if os.path.exists(manual_path) and os.path.isdir(manual_path):
            st.session_state['selected_dir'] = manual_path
            st.rerun()
        else:
            st.sidebar.error("❌ Invalid directory path")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Option 2: Browse Directories**")
    current_dir = os.path.expanduser(st.session_state.get('current_dir', home_dir))

    st.sidebar.text_area("Currently browsing:", value=current_dir, height=60, disabled=True,
                        help="Currently browsing this directory")

    if os.path.exists(current_dir) and os.path.isdir(current_dir):
        try:
            items = os.listdir(current_dir)
            dirs = sorted([d for d in items
                          if os.path.isdir(os.path.join(current_dir, d)) and not d.startswith('.')])
            dirs.insert(0, "..")
            dirs.insert(0, ".")

            selected_dir = st.sidebar.selectbox(
                "📂 Navigate to:", dirs,
                help="Select '..' to go up, '.' to use current directory, or choose a subdirectory")

            col1, col2, col3 = st.sidebar.columns(3)

            if selected_dir == ".." and col1.button("⬆️ Up", width="stretch"):
                st.session_state['current_dir'] = os.path.dirname(current_dir)
                st.rerun()
            elif selected_dir == "." and col2.button("✅ Use", width="stretch"):
                st.session_state['selected_dir'] = current_dir
                st.rerun()
            elif selected_dir not in ["..", "."]:
                if col1.button("📂 Open", width="stretch"):
                    st.session_state['current_dir'] = os.path.join(current_dir, selected_dir)
                    st.rerun()
                if col2.button("✅ Use", width="stretch"):
                    st.session_state['selected_dir'] = os.path.join(current_dir, selected_dir)
                    st.rerun()

            st.sidebar.markdown("**Quick Jump:**")
            jump_cols = st.sidebar.columns(2)
            if jump_cols[0].button("🏠 Home", width="stretch"):
                st.session_state['current_dir'] = home_dir
                st.rerun()
            if jump_cols[1].button("🖥️ Desktop", width="stretch"):
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

    return st.session_state.get('selected_dir') or current_dir

def render_node_details(data, node_labels):
    """Render node detail section."""
    selected_label = st.selectbox("Select Node", list(node_labels.values()))
    selected_node = [url for url, label in node_labels.items() if label == selected_label][0]
    node_data = next(n for n in data['nodes'] if n['url'] == selected_node)

    st.write(f"**Title:** {selected_label}")
    st.write(f"**URL:** [{node_data['url']}]({node_data['url']})")
    st.write(f"**Depth:** {node_data['depth']}")
    st.write("**Insights:**")
    with st.expander("View full insights", expanded=True):
        st.markdown(node_data['insights'])

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

# ============================================================================
# MAIN APP
# ============================================================================

st.title("🌿 Ecology Agent Graph Explorer")
st.sidebar.header("Configuration")

directory = setup_directory_browser()

if not directory or not os.path.exists(directory):
    st.error(f"❌ Please select a directory using the browser on the left")
    st.info("💡 Navigate to your graph directory and click '✅ Use Current' or '✅ Use This'")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("**Current Directory:**")
st.sidebar.code(directory, language=None)

with st.sidebar.expander("⚙️ Advanced Options"):
    file_pattern = st.text_input("File Pattern", value="*.graph_iter*.json",
                                help="Glob pattern to match JSON files")
    if st.button("🔄 Reload Files"):
        st.cache_data.clear()
        st.rerun()

with st.spinner("Loading graphs..."):
    graphs = load_graphs(directory, file_pattern)

if not graphs:
    st.error("❌ No graph files found in directory")
    st.info(f"Looking for files matching pattern: `{file_pattern}`")

    try:
        all_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        if all_files:
            st.write("**JSON files found in directory:**")
            for f in sorted(all_files)[:20]:
                st.write(f"- `{f}`")
            if len(all_files) > 20:
                st.write(f"... and {len(all_files) - 20} more")
        else:
            st.write("No JSON files found in this directory")
    except Exception as e:
        st.error(f"Error reading directory: {e}")
    st.stop()

st.sidebar.success(f"✅ Loaded {len(graphs)} iterations")
if graphs:  # Only show if graphs exist
    st.sidebar.markdown(f"**Iterations:** {min(graphs.keys())} - {max(graphs.keys())}")

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["📊 Single Graph View", "📈 Evolution Analysis", "🔎 Detailed Insights", "🧠 Topic Relationships"])

with tab1:
    st.header("Individual Graph Visualization")
    col1, col2 = st.columns([2, 1])

    with col1:
        iteration = st.select_slider("Select Iteration", options=list(graphs.keys()),
                                    value=max(graphs.keys()))
    with col2:
        layout = st.selectbox("Layout Algorithm", ['kamada', 'spring', 'circular', 'shell'], index=0)

    data = graphs[iteration]
    fig, G = create_network_graph(data, layout)
    st.plotly_chart(fig, config={'displayModeBar': True}, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes", len(data['nodes']))
    col2.metric("Edges", len(data['edges']))
    col3.metric("Max Depth", max([n['depth'] for n in data['nodes']]))
    col4.metric("Avg Degree", f"{2*len(data['edges'])/len(data['nodes']):.2f}")

    st.subheader("Node Details")
    node_labels = {n['url']: n['url'].split('/')[-1].replace('_', ' ') for n in data['nodes']}
    render_node_details(data, node_labels)

with tab2:
    st.header("Evolution Analysis")

    st.info("""
    📊 **Understanding the Metrics:**
    - **Nodes**: Number of unique topics/Wikipedia pages discovered
    - **Edges**: Number of connections/links between topics
    - **Average Depth**: Mean distance of all nodes from the starting point (how spread out the exploration is)
    - **Max Depth**: Furthest distance reached from the starting point (how deep the exploration went)
    """)

    df_metrics = compute_graph_metrics(graphs)

    st.plotly_chart(create_metric_plot(
        df_metrics, 'iteration',
        [('num_nodes', 'Nodes'), ('num_edges', 'Edges')],
        'Graph Growth Over Time', 'Count'),
        config={'displayModeBar': True}, use_container_width=True)

    st.plotly_chart(create_metric_plot(
        df_metrics, 'iteration',
        [('avg_depth', 'Average Depth'), ('max_depth', 'Max Depth')],
        'Exploration Depth Over Time', 'Depth'),
        config={'displayModeBar': True}, use_container_width=True)

    st.plotly_chart(create_metric_plot(
        df_metrics, 'iteration',
        [('density', 'Graph Density')],
        'Graph Density Over Time', 'Density'),
        config={'displayModeBar': True}, use_container_width=True)

with tab3:
    st.header("Detailed Insights Explorer")

    st.subheader("Compare Iterations")
    col1, col2 = st.columns(2)
    with col1:
        iter1 = st.selectbox("First Iteration", list(graphs.keys()), key='iter1')
    with col2:
        iter2 = st.selectbox("Second Iteration", list(graphs.keys()),
                            index=min(len(graphs)-1, 1), key='iter2')

    if iter1 != iter2:
        data1, data2 = graphs[iter1], graphs[iter2]
        urls1 = set(n['url'] for n in data1['nodes'])
        urls2 = set(n['url'] for n in data2['nodes'])
        new_nodes = urls2 - urls1

        col1, col2, col3 = st.columns(3)
        col1.metric("New Nodes", len(new_nodes))
        col2.metric("Removed Nodes", len(urls1 - urls2))
        col3.metric("Common Nodes", len(urls1 & urls2))

        if new_nodes:
            with st.expander(f"New nodes in iteration {iter2}"):
                for url in sorted(new_nodes):
                    st.write(f"- {url.split('/')[-1].replace('_', ' ')}")

    st.markdown("---")
    st.subheader("Topic Evolution")

    all_urls = set()
    for data in graphs.values():
        all_urls.update([n['url'] for n in data['nodes']])

    url_timeline = defaultdict(list)
    for iter_num, data in graphs.items():
        for node in data['nodes']:
            url_timeline[node['url']].append(iter_num)

    topic_data = [{
        'Topic': url.split('/')[-1].replace('_', ' '),
        'First Seen': min(iterations),
        'Last Seen': max(iterations),
        'Appearances': len(set(iterations))
    } for url, iterations in url_timeline.items()]

    df_topics = pd.DataFrame(topic_data).sort_values('First Seen')
    st.dataframe(df_topics, width="stretch")

    # Cumulative topics plot
    cumulative_topics = []
    seen_urls = set()
    for iter_num in sorted(graphs.keys()):
        for node in graphs[iter_num]['nodes']:
            seen_urls.add(node['url'])
        cumulative_topics.append(len(seen_urls))

    fig_cumulative = go.Figure()
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

    st.plotly_chart(fig_cumulative, config={'displayModeBar': True}, use_container_width=True)

with tab4:
    st.header("Topic Relationships (Word2Vec Analysis)")

    st.info("🧠 Training Word2Vec model on insights and topics to discover semantic relationships...")

    with st.spinner("Training Word2Vec model..."):
        w2v_model = train_word2vec_model(graphs)

    if w2v_model is None:
        st.error("❌ Could not train Word2Vec model. Not enough data or training failed.")
        st.info("💡 Make sure your graphs contain sufficient text in the insights field.")
    else:
        topic_vectors, topic_to_url = get_topic_embeddings(graphs, w2v_model)

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

            topics = list(topic_vectors.keys())
            vectors = np.array([topic_vectors[t] for t in topics])
            coords = reduce_dimensions(vectors)

            # Get starting URL and create color mapping
            starting_url = graphs[max(graphs.keys())].get('starting_url')
            starting_topic = starting_url.split('/')[-1].replace('_', ' ') if starting_url else None
            node_colors = ['red' if t == starting_topic else i for i, t in enumerate(topics)]

            embedding_fig = go.Figure(go.Scatter(
                x=coords[:, 0], y=coords[:, 1],
                mode='markers',
                hovertext=[f"<b>{t}</b><br>Index: {i}{' (Start)' if t == starting_topic else ''}"
                           for i, t in enumerate(topics)],
                marker=dict(
                    size=10,
                    color=node_colors,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Topic Index"))))

            embedding_fig.update_layout(
                title='Topic Embedding Space (t-SNE)',
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                height=600,
                hovermode='closest')

            st.plotly_chart(embedding_fig, config={'displayModeBar': True}, use_container_width=True)

            # Top topic keywords
            st.subheader("📚 Top Topic Keywords by Frequency")
            st.caption("Most frequently occurring words across all topics")

            topic_word_freq = {}
            for topic in topic_vectors.keys():
                for word in topic.lower().split():
                    if len(word) > 3:
                        topic_word_freq[word] = topic_word_freq.get(word, 0) + 1

            if topic_word_freq:
                sorted_keywords = sorted(topic_word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
                col1, col2 = st.columns(2)

                with col1:
                    keywords_df = pd.DataFrame(sorted_keywords, columns=['Word', 'Frequency'])
                    fig_keywords = go.Figure()
                    fig_keywords.add_trace(go.Bar(
                        x=keywords_df['Frequency'],
                        y=keywords_df['Word'],
                        orientation='h',
                        marker=dict(color=keywords_df['Frequency'], colorscale='Viridis', showscale=False)))

                    fig_keywords.update_layout(
                        title='Top 20 Topic Keywords',
                        xaxis_title='Frequency',
                        yaxis_title='Keyword',
                        height=500,
                        yaxis={'categoryorder': 'total ascending'})

                    st.plotly_chart(fig_keywords, config={'displayModeBar': True}, use_container_width=True)

                with col2:
                    top_keyword = sorted_keywords[0][0]
                    st.markdown(f"**Topics containing '{top_keyword}':**")
                    matching_topics = [t for t in topic_vectors.keys() if top_keyword in t.lower()]
                    for topic in matching_topics[:10]:
                        st.write(f"• {topic}")
                    if len(matching_topics) > 10:
                        st.caption(f"... and {len(matching_topics) - 10} more")

            # Topic similarity analysis
            st.subheader("Topic Similarity Analysis")

            if len(topic_vectors) > 1:
                col1, col2 = st.columns([1, 1])

                with col1:
                    topics_list = sorted(topic_vectors.keys())
                    selected_topic = st.selectbox("Select a topic to find similar topics:", topics_list)

                with col2:
                    top_k = st.slider("Number of similar topics", 5, min(20, len(topic_vectors)-1), min(10, len(topic_vectors)-1))

                if selected_topic and selected_topic in topic_vectors:
                    selected_vec = topic_vectors[selected_topic]

                    similarities = []
                    for topic, vec in topic_vectors.items():
                        if topic != selected_topic:
                            try:
                                sim = cosine_similarity([selected_vec], [vec])[0][0]
                                similarities.append((topic, float(sim)))
                            except:
                                continue

                    if similarities:
                        similarities.sort(key=lambda x: x[1], reverse=True)

                        st.markdown(f"**Most similar topics to '{selected_topic}':**")

                        top_similar = similarities[:top_k]
                        similar_topics = [t[0] for t in top_similar]
                        similar_scores = [t[1] for t in top_similar]

                        fig_sim = go.Figure()
                        fig_sim.add_trace(go.Bar(
                            x=similar_scores,
                            y=similar_topics,
                            orientation='h',
                            marker=dict(color=similar_scores, colorscale='Viridis', showscale=True)))

                        fig_sim.update_layout(
                            title=f'Top {len(top_similar)} Most Similar Topics',
                            xaxis_title='Cosine Similarity',
                            yaxis_title='Topic',
                            height=400,
                            yaxis={'categoryorder': 'total ascending'})

                        st.plotly_chart(fig_sim, config={'displayModeBar': True}, use_container_width=True)

                        with st.expander("View similarity details"):
                            df_sim = pd.DataFrame(similarities[:top_k], columns=['Topic', 'Similarity'])
                            df_sim['URL'] = df_sim['Topic'].map(topic_to_url)
                            st.dataframe(df_sim, width="stretch")
                    else:
                        st.warning("Could not calculate similarities")
            else:
                st.warning("Need at least 2 topics for similarity analysis")

            # Similarity network
            st.subheader("Topic Similarity Network")
            st.markdown("Shows topics connected when their semantic similarity exceeds the threshold")

            if len(topic_vectors) > 1:
                col1, col2 = st.columns(2)
                with col1:
                    max_topics = st.slider("Max topics to display", 10, min(100, len(topic_vectors)), min(50, len(topic_vectors)))
                with col2:
                    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.7, 0.05)

                result = create_similarity_network(topic_vectors, topic_to_url, top_n=max_topics, threshold=threshold)

                if result:
                    G_sim, sim_matrix, topics_sim = result

                    if len(G_sim.edges()) > 0:
                        pos = nx.spring_layout(G_sim, k=2, iterations=50)

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
                            marker=dict(size=15, color='lightblue', line=dict(width=2, color='white')))

                        fig_network = go.Figure(
                            data=[edge_trace, node_trace],
                            layout=go.Layout(
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=0, l=0, r=0, t=0),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                height=600))

                        st.plotly_chart(fig_network, config={'displayModeBar': True}, use_container_width=True)
                        # st.info(f"Network has {len(G_sim.nodes())} nodes and {len(G_sim.edges())} edges (threshold: {threshold})")
                    else:
                        st.warning(f"No topic pairs have similarity > {threshold}. Try lowering the threshold or adding more topics.")
                else:
                    st.warning("Need at least 2 topics for similarity network")

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
                        marker=dict(color='skyblue')))

                    fig_words.update_layout(
                        title=f"Words most similar to '{search_word}'",
                        xaxis_title='Cosine Similarity',
                        yaxis_title='Word',
                        height=400,
                        yaxis={'categoryorder': 'total ascending'})

                    st.plotly_chart(fig_words, config={'displayModeBar': True}, use_container_width=True)
                except Exception as e:
                    st.error(f"Error finding similar words: {e}")
            elif search_word:
                st.warning(f"Word '{search_word}' not found in vocabulary. Try another word.")

                try:
                    vocab_sample = list(w2v_model.wv.index_to_key[:20])
                    st.info(f"Example words in vocabulary: {', '.join(vocab_sample)}")
                except:
                    pass

st.sidebar.markdown("---")
st.sidebar.info("📊 Navigate through tabs to explore graphs, evolution, insights, and topic relationships using Word2Vec.")