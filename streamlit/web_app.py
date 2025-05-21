import streamlit as st
import requests
import networkx as nx
import matplotlib.pyplot as plt
import time
import json
import threading
import io
from PIL import Image
import matplotlib.patheffects as PathEffects
from datetime import datetime
import hashlib

# ======== Configuration ========
st.set_page_config(
    page_title="Agent State Visualizer",
    page_icon="🤖",
    layout="wide",
)

# ======== State Management ========
if 'api_data' not in st.session_state:
    st.session_state.api_data = None
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = None
if 'polling_active' not in st.session_state:
    st.session_state.polling_active = False
if 'poll_interval' not in st.session_state:
    st.session_state.poll_interval = 1.0
if 'connection_status' not in st.session_state:
    st.session_state.connection_status = {"status": "disconnected", "message": "Not connected"}
# Add state to track FSM changes
if 'fsm_hash' not in st.session_state:
    st.session_state.fsm_hash = None
if 'fsm_image' not in st.session_state:
    st.session_state.fsm_image = None

# ======== API Functions ========
def fetch_api_data(url, timeout=5):
    """Fetch data from API with error handling"""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Function to calculate hash of FSM data to detect changes
def calculate_fsm_hash(fsm_data):
    """Calculate a hash of the FSM data to detect changes"""
    if not fsm_data:
        return None

    # Convert FSM data to a stable string representation
    serialized = json.dumps(fsm_data, sort_keys=True)
    return hashlib.md5(serialized.encode('utf-8')).hexdigest()

# ======== UI Layout ========
# App title and description
st.title("Agent State Visualizer")
st.markdown("Visualize and monitor the state of the Agent FSM in real-time.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    # API endpoint configuration
    st.subheader("API Connection")
    col1, col2 = st.columns(2)
    with col1:
        api_host = st.text_input("Host", value="localhost")
    with col2:
        api_port = st.text_input("Port", value="8000")

    api_url = f"http://{api_host}:{api_port}/agent"
    st.text(f"API URL: {api_url}")

    # Polling configuration
    st.subheader("Polling Settings")
    # Store poll interval value in session state when it changes
    new_poll_interval = st.slider("Poll Interval (seconds)",
                             min_value=0.1,
                             max_value=5.0,
                             value=st.session_state.poll_interval,
                             step=0.1,
                             key="poll_interval_slider")
    st.session_state.poll_interval = new_poll_interval

    # Connection status display
    st.subheader("Connection Status")
    status_container = st.empty()

    # Display appropriate status
    status = st.session_state.connection_status.get("status", "unknown")
    message = st.session_state.connection_status.get("message", "")

    if status == "connected":
        status_container.success(message)
    elif status == "connecting":
        status_container.info(message)
    elif status == "disconnected":
        status_container.warning(message)
    elif status == "error":
        status_container.error(message)

    # Last update time
    if st.session_state.last_update_time:
        st.text(f"Last update: {st.session_state.last_update_time.strftime('%H:%M:%S')}")

    # Connection control buttons
    if st.session_state.polling_active:
        if st.button("Disconnect", key="stop_polling_button"):
            st.session_state.polling_active = False
            st.session_state.api_data = None
            st.session_state.last_update_time = None
            st.session_state.connection_status = {
                "status": "disconnected",
                "message": "Disconnected"
            }
            st.rerun()
    else:
        # Test connection button
        if st.button("Test API", key="test_api_button"):
            data = fetch_api_data(api_url)
            if "error" not in data:
                st.session_state.connection_status = {
                    "status": "connected",
                    "message": "API available"
                }
                st.session_state.api_data = data
                st.session_state.last_update_time = datetime.now()

                # Update FSM hash and image on successful test
                fsm_data = data.get("fsm", {})
                new_hash = calculate_fsm_hash(fsm_data)
                if new_hash != st.session_state.fsm_hash:
                    st.session_state.fsm_hash = new_hash
                    st.session_state.fsm_image = visualize_fsm(fsm_data)
            else:
                st.session_state.connection_status = {
                    "status": "error",
                    "message": f"API unavailable: {data.get('error')}"
                }
            st.rerun()

        # Connect button
        if st.button("Connect", key="connect_button"):
            st.session_state.polling_active = True
            st.session_state.connection_status = {
                "status": "connecting",
                "message": "Starting connection..."
            }
            st.rerun()

# Main content area - FSM Graph
st.header("Finite State Machine")
graph_container = st.empty()

# Agent info area
st.header("Agent Information")
agent_info_container = st.container()

# ======== Visualization Functions ========
def visualize_fsm(fsm_data):
    """Create a visualization of the FSM graph"""
    if not fsm_data:
        return None

    try:
        current_state = fsm_data.get("current_state")
        graph_data = fsm_data.get("graph", {})
        states = graph_data.get("states", [])
        transitions = graph_data.get("transitions", [])

        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes (states)
        for state in states:
            G.add_node(state)

        # Add edges (transitions)
        edge_labels = {}
        edge_colors = {}  # Store colors for edges based on action text

        for t in transitions:
            from_state = t.get("from")
            to_state = t.get("to")
            action = t.get("action")

            # Check if we need a new edge or need to update an existing one
            edge_key = (from_state, to_state)

            # Create a unique edge identifier if this is a multi-edge
            if edge_key in edge_labels:
                # Use a different radius for each parallel edge to avoid overlap
                # We'll modify the key slightly to create multiple edges in visualization
                i = 0
                while (from_state, to_state, i) in edge_colors:
                    i += 1
                edge_key = (from_state, to_state, i)

                # Add the edge with the unique identifier
                G.add_edge(from_state, to_state, key=i)
                edge_labels[edge_key] = action
                edge_colors[edge_key] = "red" if "failed" in action.lower() else "blue"
            else:
                # Add a new edge
                G.add_edge(from_state, to_state)
                edge_labels[edge_key] = action
                # Set color based on 'failed' in action text
                edge_colors[edge_key] = "red" if "failed" in action.lower() else "blue"

        # Create the figure
        plt.figure(figsize=(12, 8))

        # Use the original spring layout as in the source file
        pos = nx.spring_layout(G, seed=42)  # Position nodes using spring layout

        ### First draw nodes ###
        node_colors = ["lightgreen" if node == current_state else "lightblue" for node in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.8)

        # Draw node labels BELOW the nodes
        label_pos = {node: (pos[node][0], pos[node][1] - 0.1) for node in G.nodes}
        nx.draw_networkx_labels(G, label_pos, labels={node: node for node in G.nodes},
                              font_size=10, font_weight="bold", verticalalignment="top")

        # Highlight current state
        if current_state in G.nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=[current_state],
                                  node_color="green", node_size=2200, alpha=0.9)

            # Add a marker to the current state
            current_pos = pos[current_state]
            plt.plot(current_pos[0], current_pos[1], 'ro', markersize=15, alpha=0.7)
            plt.plot(current_pos[0], current_pos[1], 'ko', markersize=8)

            # Add a "Current" text label (positioned above the node)
            text = plt.text(current_pos[0], current_pos[1] - 0.1, "CURRENT",
                          horizontalalignment='center', fontsize=9, fontweight='bold')
            text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])

        ### Then draw edges ###
        drawn_edges = set()  # Track which edges we've drawn

        # First pass to calculate edge curvatures to avoid label overlap
        edge_curves = {}
        for edge_key in edge_colors:
            if len(edge_key) == 2:  # Standard edge
                from_state, to_state = edge_key
                key = 0
                rad = 0.1  # Default curvature
            else:  # Multi-edge with key
                from_state, to_state, key = edge_key
                # Use different curvatures for parallel edges
                rad = 0.1 + (key * 0.08)  # Increase curvature for each parallel edge

            edge_curves[(from_state, to_state, key)] = rad

        # Draw the edges using the calculated curvatures
        for edge_key in edge_colors:
            if len(edge_key) == 2:  # Standard edge
                from_state, to_state = edge_key
                key = 0
            else:  # Multi-edge with key
                from_state, to_state, key = edge_key

            # Get the curvature
            rad = edge_curves.get((from_state, to_state, key), 0.1)

            # Draw the edge with appropriate color and curvature
            color = edge_colors[edge_key]
            edge = [(from_state, to_state)]

            # Only draw each physical edge once
            edge_id = (from_state, to_state, key)
            if edge_id not in drawn_edges:
                nx.draw_networkx_edges(G, pos, edgelist=edge, width=2, alpha=0.7,
                                      edge_color=color, connectionstyle=f"arc3,rad={rad}",
                                      arrowsize=20, min_target_margin=20, min_source_margin=20)
                drawn_edges.add(edge_id)

        # Draw edge labels with appropriate colors and ensure they don't overlap
        for edge_key, label in edge_labels.items():
            if len(edge_key) == 2:  # Standard edge
                from_state, to_state = edge_key
                key = 0
            else:  # Multi-edge with key
                from_state, to_state, key = edge_key

            # Get the previously calculated curvature
            rad = edge_curves.get((from_state, to_state, key), 0.1)

            # Calculate label position - adjust based on the curvature to avoid overlap
            label_pos = 0.5 + (key * 0.05)  # Slightly offset each label

            # Create a dictionary with just this edge for drawing
            this_edge_label = {(from_state, to_state): label}

            # Draw the individual edge label
            nx.draw_networkx_edge_labels(G, pos, edge_labels=this_edge_label, font_size=8,
                                        font_color="black", label_pos=label_pos, alpha=0.7,
                                        bbox=dict(facecolor="white", edgecolor="none",
                                                 alpha=0.7, boxstyle="round,pad=0.2"),
                                        connectionstyle=f"arc3,rad={rad}")

        plt.title("Agent Finite State Machine", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()

        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return Image.open(buf)

    except Exception as e:
        st.error(f"Error visualizing FSM: {str(e)}")
        return None

def display_agent_info(data):
    """Display agent information including state, action, and context in a compact format"""
    if not data:
        agent_info_container.info("No agent information available.")
        return

    try:
        # Extract agent data components
        agent_name = data.get("name", "Unknown")
        agent_role = data.get("role", "Unknown")
        fsm_data = data.get("fsm", {})
        context_data = data.get("context", {})

        # Create tabs for different sections to save space
        tabs = agent_info_container.tabs(["Agent State", "Context", "Raw Data"])

        # Tab 1: Agent State
        with tabs[0]:
            col1, col2 = st.columns([1, 2])
            with col1:
                # Basic agent info in a more compact format
                st.markdown(f"**Agent:** {agent_name}")
                st.markdown(f"**Role:** {agent_role}")

                # Current state and action
                current_state = fsm_data.get("current_state", "Unknown")
                current_action = fsm_data.get("current_action", "None")
                st.markdown(f"**State:** {current_state}")
                st.markdown(f"**Action:** {current_action}")

            with col2:
                # Display available actions in a compact format
                graph_data = fsm_data.get("graph", {})
                state_actions = graph_data.get("state_actions", {})
                if current_state in state_actions:
                    actions = state_actions[current_state]
                    if actions:
                        st.markdown("**Available Actions:**")
                        # Display actions as chips/pills in a more compact visual layout
                        action_html = '<div style="display: flex; flex-wrap: wrap; gap: 5px;">'
                        for action in actions:
                            # Color-code the actions (green if no "failed" in name)
                            color = "#e0f7ea" if "failed" not in action.lower() else "#ffebee"
                            text_color = "#1b5e20" if "failed" not in action.lower() else "#c62828"
                            action_html += f'<div style="background-color: {color}; color: {text_color}; padding: 4px 8px; border-radius: 12px; font-size: 0.8em; margin-bottom: 5px;">{action}</div>'
                        action_html += '</div>'
                        st.markdown(action_html, unsafe_allow_html=True)

        # Tab 2: Context - processed to be more compact
        with tabs[1]:
            # Process context data to make it more compact
            display_context_recursive(context_data)

        # Tab 3: Raw data
        with tabs[2]:
            st.json(data)

    except Exception as e:
        agent_info_container.error(f"Error displaying agent info: {str(e)}")

def display_context_recursive(data, path="", depth=0):
    """
    Recursively display context data with collapsible sections for long text
    and nested dictionaries at any depth

    Parameters:
        data: The data to display
        path: Current path in the data structure (for unique keys)
        depth: Current recursion depth
    """
    # Base case - if not a dictionary or we've gone too deep, just show as is
    if not isinstance(data, dict) or depth > 10:  # Prevent infinite recursion
        st.json(data)
        return

    # For deeper levels, display in a single column
    for key, value in data.items():
        display_context_key(key, value, path + key + "_", depth)

def display_context_key(key, value, path, depth):
    """Display a single key-value pair with appropriate formatting for the value type"""
    # Generate a unique key for Streamlit widgets
    widget_key = f"{path}{key}_{depth}"

    # Handle various data types
    if isinstance(value, dict):
        # For dictionaries, create an expander and recurse
        with st.expander(f"**{key}**"):
            display_context_recursive(value, path, depth + 1)

    elif isinstance(value, list):
        # For lists, check the length and also contents for truncation
        list_preview = truncate_list(value)

        if len(value) > 5 or list_preview != str(value):
            # If list is long or contains long strings that were truncated
            st.markdown(f"**{key}:** *List with {len(value)} items*")
            st.markdown(f"*Preview:* {list_preview}")
            if st.checkbox(f"Show full list", key=widget_key):
                st.json(value)
        else:
            # Short list with simple values
            st.markdown(f"**{key}:** {str(value)}")

    elif isinstance(value, str):
        # For strings, check if it's long
        if len(value) > 150:
            st.markdown(f"**{key}:** *Preview:* {truncate(value)}")
            if st.checkbox(f"Show full content ({len(value)} chars)", key=widget_key):
                st.text_area("", value=value, height=300, disabled=True, key=widget_key+"_area")
        else:
            st.markdown(f"**{key}:** {value}")

    else:
        # For other types (numbers, booleans, None)
        st.markdown(f"**{key}:** {str(value)}")

def truncate(s, max_length=150):
    """Helper function to truncate long strings"""
    if isinstance(s, str) and len(s) > max_length:
        return s[:max_length] + "..."
    return s

def truncate_list(lst, max_items=5, max_str_length=50):
    """
    Helper function to truncate lists:
    - Limits number of displayed items
    - Truncates long strings within the list
    """
    if not isinstance(lst, list):
        return str(lst)

    # If list is empty or very short, just return it as is
    if len(lst) <= 3 and all(len(str(item)) <= max_str_length for item in lst):
        return str(lst)

    # Truncate the list
    preview_items = []
    for i, item in enumerate(lst[:max_items]):
        if isinstance(item, str) and len(item) > max_str_length:
            preview_items.append(f"'{item[:max_str_length]}...'")
        elif isinstance(item, dict):
            preview_items.append("{...}")
        elif isinstance(item, list):
            if len(item) > 3:
                preview_items.append(f"[{len(item)} items]")
            else:
                preview_items.append(str(item))
        else:
            preview_items.append(str(item))

    # Add ellipsis if list was truncated
    if len(lst) > max_items:
        preview_items.append(f"... ({len(lst) - max_items} more)")

    return f"[{', '.join(preview_items)}]"

# ======== Update the visualization ========
# Update the display with current data

# Simple polling mechanism (runs when active)
if st.session_state.polling_active:
    data = fetch_api_data(api_url)
    if "error" not in data:
        # Update all data first
        st.session_state.api_data = data
        st.session_state.last_update_time = datetime.now()
        st.session_state.connection_status = {
            "status": "connected",
            "message": f"Connected at {st.session_state.last_update_time.strftime('%H:%M:%S')}"
        }

        # Check if FSM data has changed
        fsm_data = data.get("fsm", {})
        new_hash = calculate_fsm_hash(fsm_data)

        # Only re-render visualization if FSM changed
        if new_hash != st.session_state.fsm_hash:
            st.session_state.fsm_hash = new_hash
            st.session_state.fsm_image = visualize_fsm(fsm_data)
            if st.session_state.fsm_image:
                graph_container.image(st.session_state.fsm_image, use_container_width=True)
            else:
                graph_container.warning("Cannot visualize FSM. Incomplete or invalid data.")
        elif st.session_state.fsm_image:
            # Use the existing visualization
            graph_container.image(st.session_state.fsm_image, use_container_width=True)
        else:
            graph_container.warning("No FSM visualization available.")

        # Display agent information (always update this)
        display_agent_info(data)
    else:
        st.session_state.connection_status = {
            "status": "error",
            "message": f"Connection error: {data.get('error')}"
        }
        graph_container.error("Error connecting to API")
        agent_info_container.error("Unable to retrieve agent information")

    # Add auto-refresh timer to keep polling
    time.sleep(st.session_state.poll_interval)  # Use the interval from session state
    st.rerun()  # Trigger a refresh to poll again

# If not polling, but we have data and an image, show it
if not st.session_state.polling_active:
    if st.session_state.api_data:
        # Display the FSM graph (use cached image if available)
        if st.session_state.fsm_image:
            graph_container.image(st.session_state.fsm_image, use_container_width=True)
        else:
            fsm_data = st.session_state.api_data.get("fsm", {})
            st.session_state.fsm_image = visualize_fsm(fsm_data)
            if st.session_state.fsm_image:
                graph_container.image(st.session_state.fsm_image, use_container_width=True)
            else:
                graph_container.warning("Cannot visualize FSM. Incomplete or invalid data.")

        # Display agent information
        display_agent_info(st.session_state.api_data)
    else:
        graph_container.info("Waiting for data from API...")
        agent_info_container.info("No agent information available")

# Footer
st.markdown("---")
st.markdown("Agent State Visualizer | Made with Streamlit")
# show_raw_data = st.checkbox("Show Raw API Data", value=False)
# if show_raw_data and st.session_state.api_data:
#     st.subheader("Raw API Data")
#     st.json(st.session_state.api_data)

