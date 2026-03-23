import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search
from google.genai import types # For HttpRetryOptions

# Load environment variables from .env file
load_dotenv()

# Configure Retry Options (from the sample code)
retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

search_agent = LlmAgent(
    name="search_agent",
    model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
    instruction="""You are a helpful search agent.
    Your primary function is to answer questions using the `google_search` tool.
    Always summarize the search results to answer the user's question.
    If the question cannot be answered by searching, state that you cannot find relevant information.
    """,
    tools=[google_search],
)