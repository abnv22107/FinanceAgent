import requests
import time
import json
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pickle
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import feedparser
from datetime import datetime, timedelta
import logging
import speech_recognition as sr
from pydub import AudioSegment

# Load API keys from .env
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")

# Initialize Groq client
groq_client = Groq(api_key=groq_api_key)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

class APIClient:
    def __init__(self, alpha_vantage_key=None):
        self.alpha_vantage_key = alpha_vantage_key or alpha_vantage_key
        self.session = requests.Session()
        
    def fetch_alpha_vantage(self, symbol, function='TIME_SERIES_INTRADAY', interval='5min'):
        if not self.alpha_vantage_key:
            raise ValueError("AlphaVantage API key is required.")
        url = 'https://www.alphavantage.co/query'
        params = {
            'function': function,
            'symbol': symbol,
            'interval': interval,
            'apikey': self.alpha_vantage_key,
            'datatype': 'json'
        }
        response = self.session.get(url, params=params)
        data = response.json()
        time.sleep(12)  # AlphaVantage API rate limit 5 requests/minute
        return data
    
    def fetch_yahoo_finance_quote(self, symbol):
        url = f'https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}'
        response = self.session.get(url)
        data = response.json()
        return data
    
    def fetch_all(self, symbol):
        results = {}
        try:
            results['alpha_vantage'] = self.fetch_alpha_vantage(symbol)
        except Exception as e:
            results['alpha_vantage_error'] = str(e)
            
        try:
            results['yahoo_finance'] = self.fetch_yahoo_finance_quote(symbol)
        except Exception as e:
            results['yahoo_finance_error'] = str(e)
            
        return results

class SECFilingsLoader:
    """Document loader for SEC filings"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.base_url = "https://www.sec.gov"
    
    def get_company_cik(self, symbol: str) -> Optional[str]:
        """Get CIK number for a stock symbol"""
        try:
            # Use SEC's company tickers JSON
            url = "https://www.sec.gov/files/company_tickers.json"
            response = self.session.get(url)
            data = response.json()
            
            for entry in data.values():
                if entry.get('ticker', '').upper() == symbol.upper():
                    cik = str(entry['cik_str']).zfill(10)
                    return cik
            return None
        except Exception as e:
            logger.error(f"Error getting CIK for {symbol}: {e}")
            return None
    
    def get_recent_filings(self, cik: str, form_types: List[str] = None, limit: int = 10) -> List[Dict]:
        """Get recent filings for a company"""
        if form_types is None:
            form_types = ['10-K', '10-Q', '8-K']
        
        try:
            # Use SEC EDGAR REST API
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            response = self.session.get(url)
            data = response.json()
            
            filings = []
            recent_filings = data.get('filings', {}).get('recent', {})
            
            if not recent_filings:
                return filings
            
            forms = recent_filings.get('form', [])
            dates = recent_filings.get('filingDate', [])
            accession_numbers = recent_filings.get('accessionNumber', [])
            
            for i, form in enumerate(forms[:50]):  # Check last 50 filings
                if form in form_types and len(filings) < limit:
                    filing_info = {
                        'form_type': form,
                        'filing_date': dates[i],
                        'accession_number': accession_numbers[i],
                        'cik': cik
                    }
                    filings.append(filing_info)
            
            return sorted(filings, key=lambda x: x['filing_date'], reverse=True)[:limit]
            
        except Exception as e:
            logger.error(f"Error getting filings for CIK {cik}: {e}")
            return []
    
    def download_filing_content(self, filing_info: Dict) -> Optional[str]:
        """Download and extract text content from a filing"""
        try:
            cik = filing_info['cik']
            accession = filing_info['accession_number'].replace('-', '')
            form_type = filing_info['form_type']
            
            # Construct filing URL
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{filing_info['accession_number']}.txt"
            
            response = self.session.get(filing_url)
            if response.status_code != 200:
                # Try alternative URL format
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{filing_info['accession_number']}-index.html"
                response = self.session.get(filing_url)
            
            if response.status_code == 200:
                content = response.text
                # Extract meaningful text content
                cleaned_content = self._clean_filing_content(content, form_type)
                return cleaned_content
            else:
                logger.warning(f"Could not download filing: {filing_url}")
                return None
                
        except Exception as e:
            logger.error(f"Error downloading filing: {e}")
            return None
    
    def _clean_filing_content(self, content: str, form_type: str) -> str:
        """Clean and extract meaningful content from SEC filing"""
        try:
            # Parse HTML content
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Extract key sections based on form type
            if form_type == '10-K':
                sections = self._extract_10k_sections(text)
                return sections
            elif form_type == '10-Q':
                sections = self._extract_10q_sections(text)
                return sections
            elif form_type == '8-K':
                return text[:5000]  # First 5000 chars for 8-K
            
            return text[:10000]  # First 10000 chars as fallback
            
        except Exception as e:
            logger.error(f"Error cleaning filing content: {e}")
            return content[:5000]  # Return first 5000 chars as fallback
    
    def _extract_10k_sections(self, text: str) -> str:
        """Extract key sections from 10-K filing"""
        sections_to_find = [
            "ITEM 1. BUSINESS",
            "ITEM 1A. RISK FACTORS", 
            "ITEM 2. PROPERTIES",
            "ITEM 3. LEGAL PROCEEDINGS",
            "ITEM 7. MANAGEMENT'S DISCUSSION"
        ]
        
        extracted_sections = []
        text_upper = text.upper()
        
        for section in sections_to_find:
            start_idx = text_upper.find(section)
            if start_idx != -1:
                # Find next section or end
                end_idx = len(text)
                for other_section in sections_to_find:
                    if other_section != section:
                        other_idx = text_upper.find(other_section, start_idx + len(section))
                        if other_idx != -1 and other_idx < end_idx:
                            end_idx = other_idx
                
                section_text = text[start_idx:start_idx + min(3000, end_idx - start_idx)]
                extracted_sections.append(f"{section}:\n{section_text}")
        
        return "\n\n".join(extracted_sections) if extracted_sections else text[:10000]
    
    def _extract_10q_sections(self, text: str) -> str:
        """Extract key sections from 10-Q filing"""
        sections_to_find = [
            "ITEM 2. MANAGEMENT'S DISCUSSION",
            "ITEM 4. CONTROLS AND PROCEDURES"
        ]
        
        extracted_sections = []
        text_upper = text.upper()
        
        for section in sections_to_find:
            start_idx = text_upper.find(section)
            if start_idx != -1:
                section_text = text[start_idx:start_idx + 2000]
                extracted_sections.append(f"{section}:\n{section_text}")
        
        return "\n\n".join(extracted_sections) if extracted_sections else text[:5000]

class ScrapingAgent:
    """Agent for scraping SEC filings and other financial documents"""
    
    def __init__(self):
        self.sec_loader = SECFilingsLoader()
        self.session = requests.Session()
    
    def scrape_company_filings(self, symbol: str, form_types: List[str] = None, limit: int = 5) -> List[DocumentChunk]:
        """Scrape recent SEC filings for a company"""
        if form_types is None:
            form_types = ['10-K', '10-Q', '8-K']
        
        logger.info(f"Scraping SEC filings for {symbol}")
        chunks = []
        
        # Get CIK for the symbol
        cik = self.sec_loader.get_company_cik(symbol)
        if not cik:
            logger.warning(f"Could not find CIK for symbol {symbol}")
            return chunks
        
        # Get recent filings
        filings = self.sec_loader.get_recent_filings(cik, form_types, limit)
        
        for filing in filings:
            logger.info(f"Processing {filing['form_type']} filing from {filing['filing_date']}")
            
            # Download filing content
            content = self.sec_loader.download_filing_content(filing)
            
            if content:
                # Split content into smaller chunks
                chunk_size = 2000
                content_chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                
                for i, chunk_content in enumerate(content_chunks):
                    if len(chunk_content.strip()) > 100:  # Only add meaningful chunks
                        chunk = DocumentChunk(
                            content=chunk_content.strip(),
                            metadata={
                                'symbol': symbol,
                                'source': 'sec_filing',
                                'form_type': filing['form_type'],
                                'filing_date': filing['filing_date'],
                                'chunk_index': i,
                                'total_chunks': len(content_chunks),
                                'cik': cik
                            }
                        )
                        chunks.append(chunk)
            
            # Add delay to be respectful to SEC servers
            time.sleep(1)
        
        logger.info(f"Scraped {len(chunks)} chunks from SEC filings for {symbol}")
        return chunks
    
    def scrape_financial_news(self, symbol: str, limit: int = 5) -> List[DocumentChunk]:
        """Scrape recent financial news for a company"""
        chunks = []
        
        try:
            # Yahoo Finance RSS feed
            rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
            feed = feedparser.parse(rss_url)
            
            for entry in feed.entries[:limit]:
                content = f"""
                Title: {entry.title}
                Published: {entry.published}
                Summary: {entry.get('summary', 'No summary available')}
                Link: {entry.link}
                """
                
                chunk = DocumentChunk(
                    content=content.strip(),
                    metadata={
                        'symbol': symbol,
                        'source': 'financial_news',
                        'title': entry.title,
                        'published': entry.published,
                        'link': entry.link
                    }
                )
                chunks.append(chunk)
        
        except Exception as e:
            logger.error(f"Error scraping news for {symbol}: {e}")
        
        return chunks

class FinancialDataProcessor:
    """Process financial data into structured text chunks"""
    
    @staticmethod
    def process_alpha_vantage_data(data: Dict, symbol: str) -> List[DocumentChunk]:
        chunks = []
        
        if 'Time Series (5min)' in data:
            time_series = data['Time Series (5min)']
            for timestamp, values in list(time_series.items())[:20]:  # Limit to recent 20 entries
                content = f"""
                Stock: {symbol}
                Timestamp: {timestamp}
                Open: ${values.get('1. open', 'N/A')}
                High: ${values.get('2. high', 'N/A')}
                Low: ${values.get('3. low', 'N/A')}
                Close: ${values.get('4. close', 'N/A')}
                Volume: {values.get('5. volume', 'N/A')}
                Source: Alpha Vantage
                """.strip()
                
                chunks.append(DocumentChunk(
                    content=content,
                    metadata={
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'source': 'alpha_vantage',
                        'data_type': 'time_series'
                    }
                ))
        
        return chunks
    
    @staticmethod
    def process_yahoo_finance_data(data: Dict, symbol: str) -> List[DocumentChunk]:
        chunks = []
        
        if 'quoteResponse' in data and 'result' in data['quoteResponse']:
            for quote in data['quoteResponse']['result']:
                content = f"""
                Stock: {quote.get('symbol', symbol)}
                Company: {quote.get('longName', 'N/A')}
                Current Price: ${quote.get('regularMarketPrice', 'N/A')}
                Previous Close: ${quote.get('previousClose', 'N/A')}
                Market Cap: {quote.get('marketCap', 'N/A')}
                PE Ratio: {quote.get('trailingPE', 'N/A')}
                52 Week High: ${quote.get('fiftyTwoWeekHigh', 'N/A')}
                52 Week Low: ${quote.get('fiftyTwoWeekLow', 'N/A')}
                Volume: {quote.get('regularMarketVolume', 'N/A')}
                Source: Yahoo Finance
                """.strip()
                
                chunks.append(DocumentChunk(
                    content=content,
                    metadata={
                        'symbol': symbol,
                        'source': 'yahoo_finance',
                        'data_type': 'quote'
                    }
                ))
        
        return chunks

class QueryAnalyzer:
    """Analyzes queries to determine which data sources and functions to use"""
    
    def __init__(self):
        self.source_patterns = {
            'market_data': [
                r'price', r'stock price', r'current price', r'market price',
                r'volume', r'trading volume', r'market cap', r'pe ratio',
                r'52 week', r'high', r'low'
            ],
            'sec_filings': [
                r'sec filing', r'10-k', r'10-q', r'8-k', r'annual report',
                r'quarterly report', r'risk factor', r'business development',
                r'management discussion', r'financial statement'
            ],
            'news': [
                r'news', r'recent news', r'latest news', r'update',
                r'development', r'announcement', r'report'
            ]
        }
        
    def analyze_query(self, query: str) -> Dict[str, bool]:
        """Analyze query to determine which data sources to use"""
        query = query.lower()
        sources = {
            'market_data': False,
            'sec_filings': False,
            'news': False
        }
        
        for source, patterns in self.source_patterns.items():
            if any(re.search(pattern, query) for pattern in patterns):
                sources[source] = True
                
        # If no specific patterns match, use all sources
        if not any(sources.values()):
            sources = {k: True for k in sources}
            
        return sources

class RetrieverAgent:
    def __init__(self, embedding_model='all-MiniLM-L6-v2', index_path='financial_index.faiss'):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index_path = index_path
        self.chunks_path = index_path.replace('.faiss', '_chunks.pkl')
        self.index = None
        self.chunks = []
        self.api_client = APIClient(alpha_vantage_key)
        self.data_processor = FinancialDataProcessor()
        self.scraping_agent = ScrapingAgent()
        self.query_analyzer = QueryAnalyzer()
        
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts"""
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def fetch_relevant_data(self, symbol: str, sources: Dict[str, bool]) -> List[DocumentChunk]:
        """Fetch data only from relevant sources based on query analysis"""
        chunks = []
        
        if sources['market_data']:
            # Fetch market data
            raw_data = self.api_client.fetch_all(symbol)
            if 'alpha_vantage' in raw_data:
                chunks.extend(self.data_processor.process_alpha_vantage_data(
                    raw_data['alpha_vantage'], symbol
                ))
            if 'yahoo_finance' in raw_data:
                chunks.extend(self.data_processor.process_yahoo_finance_data(
                    raw_data['yahoo_finance'], symbol
                ))
        
        if sources['sec_filings']:
            # Fetch SEC filings
            chunks.extend(self.scraping_agent.scrape_company_filings(symbol))
            
        if sources['news']:
            # Fetch news
            chunks.extend(self.scraping_agent.scrape_financial_news(symbol))
            
        return chunks
    
    def build_index(self, symbols: List[str], include_filings: bool = True, include_news: bool = True):
        """Build FAISS index from financial data for given symbols"""
        print(f"Building index for symbols: {symbols}")
        all_chunks = []
        
        # Use all sources for initial index building
        sources = {
            'market_data': True,
            'sec_filings': include_filings,
            'news': include_news
        }
        
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            chunks = self.fetch_relevant_data(symbol, sources)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            raise ValueError("No data chunks created. Check your API keys and data sources.")
        
        # Create embeddings
        texts = [chunk.content for chunk in all_chunks]
        embeddings = self.create_embeddings(texts)
        
        # Add embeddings to chunks
        for i, chunk in enumerate(all_chunks):
            chunk.embedding = embeddings[i]
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        self.chunks = all_chunks
        
        # Save index and chunks
        faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"Index built with {len(all_chunks)} chunks")
    
    def load_index(self):
        """Load existing FAISS index and chunks"""
        if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            print(f"Loaded index with {len(self.chunks)} chunks")
            return True
        return False
    
    def retrieve(self, query: str, k: int = 5) -> List[DocumentChunk]:
        """Retrieve top-k most relevant chunks for a query"""
        if self.index is None:
            raise ValueError("Index not built or loaded. Call build_index() or load_index() first.")
        
        # Create query embedding
        query_embedding = self.create_embeddings([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Return relevant chunks
        relevant_chunks = []
        for i, score in zip(indices[0], scores[0]):
            if i < len(self.chunks):
                chunk = self.chunks[i]
                chunk.metadata['relevance_score'] = float(score)
                relevant_chunks.append(chunk)
        
        return relevant_chunks
    
    def query_with_context(self, query: str, k: int = 5) -> str:
        """Retrieve relevant chunks and generate response using Groq"""
        # Analyze query to determine which sources to use
        sources = self.query_analyzer.analyze_query(query)
        
        # Extract symbols from query if present
        symbols = re.findall(r'\b[A-Z]{1,5}\b', query)  # Basic symbol detection
        if symbols:
            # Refresh data for mentioned symbols
            for symbol in symbols:
                new_chunks = self.fetch_relevant_data(symbol, sources)
                if new_chunks:
                    # Update index with new chunks
                    texts = [chunk.content for chunk in new_chunks]
                    embeddings = self.create_embeddings(texts)
                    faiss.normalize_L2(embeddings)
                    self.index.add(embeddings)
                    self.chunks.extend(new_chunks)
        
        # Get relevant chunks
        relevant_chunks = self.retrieve(query, k)
        
        # Prepare context
        context = "\n\n".join([
            f"Chunk {i+1} (Score: {chunk.metadata.get('relevance_score', 0):.3f}) - Source: {chunk.metadata.get('source', 'unknown')}:\n{chunk.content}"
            for i, chunk in enumerate(relevant_chunks)
        ])
        
        # Create prompt
        prompt = f"""
        You are a financial analyst AI. Use the following financial data to answer the user's question.
        
        Context:
        {context}
        
        Question: {query}
        
        Please provide a comprehensive answer based on the provided financial data including market data, SEC filings, and news. If the data is insufficient, mention what additional information would be helpful.
        """
        
        # Generate response using Groq
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a helpful financial analyst AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

def prepare_voice_file(path: str) -> str:
    """
    Converts the input audio file to WAV format if necessary and returns the path to the WAV file.
    """
    if os.path.splitext(path)[1] == '.wav':
        return path
    elif os.path.splitext(path)[1] in ('.mp3', '.m4a', '.ogg', '.flac'):
        audio_file = AudioSegment.from_file(
            path, format=os.path.splitext(path)[1][1:])
        wav_file = os.path.splitext(path)[0] + '.wav'
        audio_file.export(wav_file, format='wav')
        return wav_file
    else:
        raise ValueError(
            f'Unsupported audio format: {format(os.path.splitext(path)[1])}')

def transcribe_audio(audio_data, language) -> str:
    """
    Transcribes audio data to text using Google's speech recognition API.
    """
    r = sr.Recognizer()
    text = r.recognize_google(audio_data, language=language)
    return text

def process_voice_input(input_path: str, language: str = 'en-US') -> str:
    """
    Processes a voice input file and returns the transcribed text.
    """
    try:
        wav_file = prepare_voice_file(input_path)
        with sr.AudioFile(wav_file) as source:
            audio_data = sr.Recognizer().record(source)
            text = transcribe_audio(audio_data, language)
            print('Transcribed text:', text)
            return text
    except Exception as e:
        print('Error processing voice input:', e)
        return None

def main():
    # Initialize retriever agent
    retriever = RetrieverAgent()
    
    # Try to load existing index, or build new one
    if not retriever.load_index():
        print("Building new index...")
        symbols = ['AAPL', 'MSFT']  # Example symbols (reduced for faster testing)
        retriever.build_index(symbols, include_filings=True, include_news=True)
    
    print("\nWelcome to the Financial Analysis Chat!")
    print("You can ask questions about stocks, SEC filings, and financial news.")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'help' to see example questions.")
    print("Type 'voice' to switch to voice input mode.")
    print("=" * 70)

    input_mode = "text"  # Default input mode

    while True:
        try:
            if input_mode == "text":
                # Get user input
                query = input("\nYour question: ").strip()
                
                # Check for mode switch
                if query.lower() == 'voice':
                    input_mode = "voice"
                    print("\nSwitched to voice input mode. Please provide the path to your audio file.")
                    continue
            else:
                # Voice input mode
                print("\nPlease enter the path to your audio file (or type 'text' to switch back):")
                input_path = input().strip()
                
                if input_path.lower() == 'text':
                    input_mode = "text"
                    print("\nSwitched to text input mode.")
                    continue
                
                if not os.path.isfile(input_path):
                    print('Error: File not found.')
                    continue
                
                query = process_voice_input(input_path)
                if not query:
                    continue
            
            # Check for exit command
            if query.lower() in ['exit', 'quit', 'bye']:
                print("\nThank you for using the Financial Analysis Chat. Goodbye!")
                break
                
            # Check for help command
            if query.lower() == 'help':
                print("\nExample questions you can ask:")
                print("1. What are the recent business developments for Apple according to SEC filings?")
                print("2. What is the current price of Apple stock?")
                print("3. Show me the latest news about Microsoft")
                print("4. What are the risk factors mentioned in recent 10-K filings?")
                print("5. Compare the financial performance of these companies")
                continue
                
            # Skip empty queries
            if not query:
                continue
                
            print("\nAnalyzing your question...")
            
            # Get relevant chunks
            chunks = retriever.retrieve(query, k=5)
            print(f"Found {len(chunks)} relevant chunks from sources: {[c.metadata.get('source') for c in chunks]}")
            
            # Generate response with context
            response = retriever.query_with_context(query, k=5)
            print(f"\nResponse: {response}\n")
            print("=" * 70)
            
        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try asking your question differently or type 'help' for examples.")

if __name__ == "__main__":
    main()