
# Local RAG Agent with LLaMA3

This project demonstrates how to build a Retrieval-Augmented Generation (RAG) agent using local models, search tools, and LangChain.

## Key Concepts

- **Routing:** Adaptively route questions to a vectorstore or web search based on relevance.
- **Fallback:** Use corrective measures to fallback to web search when relevant documents are not retrieved.
- **Self-correction:** Implement checks to ensure the answers are grounded and relevant, correcting hallucinations.

## Components Overview

### 1. **Local Models**
- Embeddings:
  - Nomic Embeddings using GPT4All.
- LLM:
  - Ollama with the Llama 3.2 model (`llama3.2:3b-instruct-fp16`) for generating responses.

### 2. **Search**
- **Tavily:** A search engine optimized for LLMs and RAG tasks.
- **LangSmith:** Optional tracing is available for detailed workflow monitoring.

### 3. **Document Loading & Splitting**
Documents from URLs are loaded using `WebBaseLoader` and split into chunks using `RecursiveCharacterTextSplitter`.

### 4. **Vectorstore & Embeddings**
Documents are embedded using `NomicEmbeddings` and stored in an SKLearn vector database.

### 5. **Retriever**
The retriever fetches the top `k` documents from the vector database that match the user query.

### 6. **Router**
A custom router that sends queries to either the vector database or a web search based on the question's context.

```python
router_instructions = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search.
"""
```

### 7. **Grading System**
- **Document Relevance:** Checks if a document is relevant to the user question.
- **Hallucination Check:** Ensures the generated answer is grounded in facts from the documents.
- **Answer Quality:** Grades the generated answer to check if it fully addresses the user's query.

### 8. **Graph Workflow**
The workflow is implemented using LangGraph, where nodes represent different steps:
- Retrieval
- Document relevance grading
- Answer generation
- Hallucination checking
- Web search integration

```python
def route_question(state):
    """Route question to web search or RAG."""
    # logic for routing to vectorstore or web search
    return "websearch" if some_condition else "vectorstore"
```

## Example Usage

```python
inputs = {"question": "What are the types of agent memory?", "max_retries": 3}
for event in graph.stream(inputs, stream_mode="values"):
    print(event)
```

## Traces

You can view traces for both workflows on LangSmith:

- [RAG Agent Trace 1](https://smith.langchain.com/public/1e01baea-53e9-4341-a6d1-b1614a800a97/r)
- [RAG Agent Trace 2](https://smith.langchain.com/public/acdfa49d-aa11-48fb-9d9c-13a687ff311f/r)

## Installation

Run the following command to install the necessary packages:

```bash
pip install -U langchain langchain_community tiktoken langchain-nomic "nomic[local]" langchain-ollama scikit-learn langgraph tavily-python bs4
```

## Running the Code

Ensure you have the necessary environment variables set for Tavily and LangSmith:

```python
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("TAVILY_API_KEY")
_set_env("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "local-llama32-rag"
```

### Load the model

```python
from langchain_ollama import ChatOllama
local_llm = 'llama3.2:3b-instruct-fp16'
llm = ChatOllama(model=local_llm, temperature=0)
```

### Load and split documents

```python
from langchain_community.document_loaders import WebBaseLoader
urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/", "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"]
docs = [WebBaseLoader(url).load() for url in urls]
```

## Conclusion

This project demonstrates how to combine local LLM models, adaptive RAG techniques, and web search to build a self-correcting, high-quality retrieval-augmented question-answering agent.
