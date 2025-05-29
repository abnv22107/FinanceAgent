# FastAPI Services for All Agents

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional

from tools.api_tool import api_tool  # Wraps APIClient logic
from tools.scraper_tool import scraper_tool  # Wraps ScrapingAgent logic
from tools.analytics_tool import analytics_tool  # Wraps analytics logic
from tools.retriever_tool import retriever_tool  # Wraps RetrieverAgent logic
from tools.voice_tool import voice_tool  # Wraps voice STT/TTS logic

app = FastAPI()

# --------------------- API Agent ---------------------
class APISymbolRequest(BaseModel):
    symbol: str

@app.post("/api/market-data")
def get_market_data(request: APISymbolRequest):
    return api_tool.fetch_all(request.symbol)

# --------------------- Scraper Agent ---------------------
class FilingRequest(BaseModel):
    symbol: str
    form_types: Optional[List[str]] = ["10-K", "10-Q", "8-K"]
    limit: Optional[int] = 5

@app.post("/scrape/filings")
def scrape_filings(request: FilingRequest):
    return scraper_tool.scrape_company_filings(request.symbol, request.form_types, request.limit)

@app.post("/scrape/news")
def scrape_news(request: APISymbolRequest):
    return scraper_tool.scrape_financial_news(request.symbol)

# --------------------- Retriever Agent ---------------------
class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 5

@app.post("/query")
def query_retriever(request: QueryRequest):
    return retriever_tool.query_with_context(request.query, k=request.k)

class IndexRequest(BaseModel):
    symbols: List[str]
    include_filings: Optional[bool] = True
    include_news: Optional[bool] = True

@app.post("/update-index")
def update_index(request: IndexRequest):
    retriever_tool.build_index(request.symbols, request.include_filings, request.include_news)
    return {"status": "Index updated."}

# --------------------- Analytics Agent ---------------------
class ExposureRequest(BaseModel):
    holdings: List[dict]  # Each dict should have symbol and amount

@app.post("/analyze/exposure")
def analyze_exposure(request: ExposureRequest):
    return analytics_tool.analyze_portfolio(request.holdings)

# --------------------- Voice Agent ---------------------
class VoiceTranscriptionRequest(BaseModel):
    file_path: str
    language: Optional[str] = 'en-US'

@app.post("/voice/stt")
def transcribe_audio(request: VoiceTranscriptionRequest):
    return {"text": voice_tool.process_voice_input(request.file_path, request.language)}

class TextToSpeechRequest(BaseModel):
    text: str
    voice: Optional[str] = 'default'

@app.post("/voice/tts")
def convert_text_to_speech(request: TextToSpeechRequest):
    return voice_tool.convert_to_speech(request.text, request.voice)
