from main import RetrieverAgent

# Initialize the retriever with default parameters
retriever_tool = RetrieverAgent(
    embedding_model='all-MiniLM-L6-v2',
    index_path='financial_index.faiss'
)

# Build initial index with some common stocks
initial_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
retriever_tool.build_index(initial_symbols, include_filings=True, include_news=True) 