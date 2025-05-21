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

    api_url = f"http://{api_host}:{api_port}/state"
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
        for t in transitions:
            from_state = t.get("from")
            to_state = t.get("to")
            action = t.get("action")

            # Add the edge
            if (from_state, to_state) not in G.edges:
                G.add_edge(from_state, to_state)
                edge_labels[(from_state, to_state)] = action
            else:
                # If multiple actions between same states, combine them
                existing_label = edge_labels.get((from_state, to_state), "")
                edge_labels[(from_state, to_state)] = f"{existing_label}\n{action}" if existing_label else action

        # Create the figure
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)  # Position nodes using spring layout

        # Draw nodes
        node_colors = ["lightgreen" if node == current_state else "lightblue" for node in G.nodes]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.8)

        # Draw node labels
        node_labels = {node: node for node in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight="bold")

        # Draw edges
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color="gray",
                               connectionstyle="arc3,rad=0.1", arrowsize=20)

        # Draw edge labels
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8,
                                    font_color="red", label_pos=0.3, alpha=0.7)

        # Highlight current state
        if current_state in G.nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=[current_state],
                                  node_color="green", node_size=2200, alpha=0.9)

            # Add a marker to the current state
            current_pos = pos[current_state]
            plt.plot(current_pos[0], current_pos[1], 'ro', markersize=15, alpha=0.7)
            plt.plot(current_pos[0], current_pos[1], 'ko', markersize=8)

            # Add a "Current" text label
            text = plt.text(current_pos[0], current_pos[1] - 0.1, "CURRENT",
                          horizontalalignment='center', fontsize=9, fontweight='bold')
            text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])

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
    """Display agent information including state, action, and context"""
    if not data:
        agent_info_container.info("No agent information available.")
        return

    try:
        # Extract agent data components
        agent_name = data.get("name", "Unknown")
        agent_role = data.get("role", "Unknown")
        fsm_data = data.get("fsm", {})
        context_data = data.get("context", {})

        # Display agent name and basic info
        agent_info_container.subheader(f"Agent: {agent_name}")

        # Create columns for state and context
        col1, col2 = agent_info_container.columns(2)

        # Display current state and action
        with col1:
            st.markdown("### Current State")
            current_state = fsm_data.get("current_state", "Unknown")
            current_action = fsm_data.get("current_action", "None")
            st.markdown(f"**State:** {current_state}")
            st.markdown(f"**Action:** {current_action}")

            # Display available actions
            graph_data = fsm_data.get("graph", {})
            state_actions = graph_data.get("state_actions", {})
            if current_state in state_actions:
                actions = state_actions[current_state]
                if actions:
                    st.markdown("**Available Actions:**")
                    for action in actions:
                        st.markdown(f"- {action}")

        # Display agent context
        with col2:
            st.markdown("### Context")

            # Make a copy and truncate large content for display
            context_copy = json.loads(json.dumps(context_data))
            if 'selected_file' in context_copy and 'content' in context_copy['selected_file']:
                content = context_copy['selected_file']['content']
                if len(content) > 300:
                    context_copy['selected_file']['content'] = content[:300] + "... [truncated]"

            # Display the context
            st.json(context_copy)

    except Exception as e:
        agent_info_container.error(f"Error displaying agent info: {str(e)}")

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
show_raw_data = st.checkbox("Show Raw API Data", value=False)
if show_raw_data and st.session_state.api_data:
    st.subheader("Raw API Data")
    st.json(st.session_state.api_data)

st.markdown("Agent State Visualizer | Made with Streamlit")
