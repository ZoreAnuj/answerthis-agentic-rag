# AnswerThis Agentic RAG Engine

An Agentic AI–RAG engine designed to deliver fast, accurate answers by intelligently combining internal knowledge with real-time web search. It explores the architecture for a responsive assistant that can decide when to use its own data versus fetching fresh information.

## Key Features
*   **Hybrid Retrieval:** Seamlessly queries internal documents or performs a Google search based on the user's question.
*   **Agentic Decision-Making:** An LLM-powered router analyzes queries to determine the optimal data source.
*   **Streamlit Interface:** Provides a clean, interactive web UI for testing and demonstration.
*   **Modular Design:** Separates core logic, agent routing, and UI components for clarity.

## Tech Stack
*   **Backend:** Python, LangChain, LangGraph
*   **LLM:** OpenAI GPT
*   **Search:** Google Serper API
*   **Vector Store:** ChromaDB
*   **Frontend:** Streamlit

## Getting Started
1.  Clone the repo: `git clone https://github.com/zoreanuj/answerthis-agentic-rag.git`
2.  Install dependencies: `pip install -r requirements.txt`
3.  Set your API keys in `.env`: `OPENAI_API_KEY`, `SERPER_API_KEY`
4.  Run the app: `streamlit run app.py`