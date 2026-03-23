import asyncio
import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool
from google.genai import types
import google.generativeai as genai
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

from src.agents.rag_agent import rag_agent
from src.agents.search_agent import search_agent

# --- Initialization ---
# Load environment variables
load_dotenv()

# Configure Google Generative AI
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY must be set in the .env file.")
genai.configure(api_key=api_key)

# --- Agent and App Setup ---
# Configure retry options
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],
)

# Define the orchestrator agent
orchestrator_agent = LlmAgent(
    name="OrchestratorAgent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction="""You are an expert orchestrator designed to answer user queries comprehensively.
    First, attempt to answer the question using the `rag_agent`.
    If the `rag_agent` indicates it cannot answer the question (e.g., by returning an empty string or explicitly stating lack of information),
    then use the `search_agent` to find relevant information from Google.
    Combine information from both sources if available and relevant, prioritizing factual information from the RAG agent when present.
    Always provide a detailed and helpful answer.
    """,
    tools=[
        AgentTool(agent=rag_agent),
        AgentTool(agent=search_agent)
    ],
)

# Initialize the agent runner
runner = InMemoryRunner(agent=orchestrator_agent)

# --- FastAPI Application ---
app = FastAPI(
    title="Agentic AI System",
    description="An agentic AI system with RAG and Google Search capabilities.",
    version="1.0.0",
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    
def get_final_text_response(response_events: list) -> str:
    """Extracts the final text response from the agent's execution events."""
    final_text = ""
    for event in reversed(response_events):
        if hasattr(event.content, "parts") and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    final_text = part.text
                    return final_text # Return the last text response found
    return final_text if final_text else "No text response found."

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """
    Accepts a user query, runs it through the orchestrator agent,
    and returns the final response.
    """
    print(f"Received query: {request.query}")
    
    # Run the agent with the provided query
    response_events = await runner.run_debug(request.query)
    
    # Extract the final text response from the event trace
    final_response = get_final_text_response(response_events)
    
    print(f"Final response: {final_response}")
    return {"response": final_response}

# To run the app, save this as main.py and run: uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
