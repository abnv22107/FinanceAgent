import streamlit as st
import time
from app import RetrieverAgent
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Page config
st.set_page_config(
    page_title="Financial Assistant AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        gap: 0.5rem;
    }
    .chat-message.user {
        background-color: #2b313e;
        color: white;
    }
    .chat-message.assistant {
        background-color: #f0f2f6;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .message {
        width: 80%;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'retriever' not in st.session_state:
    with st.spinner('Initializing AI...'):
        st.session_state.retriever = RetrieverAgent()
        if not st.session_state.retriever.load_index():
            st.info('Building initial index with common tech stocks...')
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
            st.session_state.retriever.build_index(symbols, include_filings=True, include_news=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("📊 Financial Assistant AI")
    st.markdown("---")
    
    st.markdown("### About")
    st.markdown("""
    This AI-powered financial analysis tool can:
    - Analyze stock prices and trends
    - Review SEC filings
    - Summarize financial news
    - Provide market insights
    """)
    
    st.markdown("### Example Queries")
    st.markdown("""
    - What is the current price of Apple stock?
    - What are the recent business developments for Microsoft?
    - Show me the latest news about Google
    - What are the risk factors mentioned in recent 10-K filings?
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ by Financial Assistant AI")

# Main content
st.title("🤖 Financial Assistant Chat")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about financial markets..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                # Get relevant chunks
                chunks = st.session_state.retriever.retrieve(prompt, k=5)
                
                # Generate response
                response = st.session_state.retriever.query_with_context(prompt, k=5)
                
                # Display response
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # If the query is about stock prices, add a chart
                if any(word in prompt.lower() for word in ['price', 'stock', 'chart', 'trend']):
                    try:
                        # Create a sample price chart (you can replace this with real data)
                        dates = pd.date_range(end=datetime.now(), periods=30)
                        prices = pd.Series(
                            [100 + i * 2 + np.random.normal(0, 1) for i in range(30)],
                            index=dates
                        )
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=dates,
                            y=prices,
                            mode='lines',
                            name='Stock Price',
                            line=dict(color='#00ff00', width=2)
                        ))
                        
                        fig.update_layout(
                            title='30-Day Price Trend',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            template='plotly_dark',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning("Could not generate price chart.")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please try rephrasing your question.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Powered by Advanced AI Technology | Data from SEC, Alpha Vantage, and Yahoo Finance</p>
</div>
""", unsafe_allow_html=True) 