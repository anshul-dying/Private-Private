# LLM Query Retrieval System

A sophisticated AI-powered system designed to intelligently process and answer queries from insurance policy documents. Built for the Bajaj Finserv Hackathon, this system combines advanced natural language processing with document analysis to provide accurate, context-aware responses to insurance-related questions.

## What is this project?

This is an intelligent document query system that:

- **Processes Insurance Documents**: Automatically extracts and analyzes text from PDF insurance policy documents
- **Semantic Search**: Uses advanced embedding technology to find the most relevant document sections for any query
- **Intelligent Answering**: Leverages Large Language Models (LLMs) to generate accurate, context-aware responses
- **RESTful API**: Provides a clean API interface for document processing and query handling
- **Analytics**: Tracks query patterns and provides insights into document usage

The system is specifically designed for insurance policy documents but can be adapted for other document types. It handles complex queries by matching them against relevant document clauses and generating human-like responses.

## How to Setup this Project

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Git

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd llm-query-retrieval
   ```

2. **Create a virtual environment**
   ```bash
   # Using python
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate

   # Using Anaconda
   conda create -p venv python==3.10
   conda activate ./venv
   ```


3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the root directory with the following variables:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   OPENROUTER_REFERER=https://github.com/your-repo
   USE_LOCAL_LLM=true
   LOCAL_LLM_URL=http://localhost:11434/api/generate
   LOCAL_LLM_MODEL=llama3.2:3b
   DATABASE_URL=sqlite:///database.db
   LOG_LEVEL=INFO
   ```

5. **Optional: Set up local LLM (Ollama)**
   
   If you want to use a local LLM instead of cloud services:
   ```bash
   # Install Ollama (https://ollama.ai/)
   # Then pull the model
   ollama pull llama3.2:3b
   # Then start model API
   ollama serve
   # Will run on port 11434
   ```

6. **Run the application**
   ```bash
   # Start the FastAPI server
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

7. **Access the API**
   
   The API will be available at `http://localhost:8000`

### Project Structure

```
├── api/                    # FastAPI application
│   ├── main.py            # Main application entry point
│   ├── routes/            # API route definitions
│   └── models/            # Pydantic models
├── core/                  # Core business logic
│   ├── decision_engine.py # Main decision making logic
│   ├── llm_client.py      # LLM integration
│   ├── embedding_generator.py # Text embedding generation
│   ├── document_processor.py # Document processing
│   ├── clause_matcher.py  # Clause matching logic
│   ├── predefined_answers.py # Predefined Q&A handling
│   └── logger_manager.py  # Logging management
├── config/                # Configuration files
├── database/              # Database related files
├── Docs/                  # Sample documents and queries
├── scripts/               # Utility scripts
├── tests/                 # Test files
└── requirements.txt       # Python dependencies
```

## How it Works

The system operates through several sophisticated components working together:

### 1. Document Processing
- **Text Extraction**: Documents are processed to extract text content using PyPDF2 and python-docx
- **Chunking**: Large documents are broken down into manageable clauses for better processing
- **Embedding Generation**: Each clause is converted into high-dimensional vector embeddings using sentence-transformers
- **Storage**: Document metadata and embeddings are stored in a SQLite database with FAISS indexing for fast retrieval

### 2. Query Processing
When a query is received, the system follows this workflow:

1. **Predefined Answer Check**: First checks if the query matches any predefined answers for instant responses
2. **Semantic Search**: Uses FAISS vector similarity search to find the most relevant document clauses
3. **Context Assembly**: Combines the most relevant clauses to provide context for the LLM
4. **Response Generation**: Sends the query and context to the LLM for answer generation

### 3. Intelligent Answering
The system employs multiple strategies for generating responses:

- **Predefined Answers**: For common questions, uses pre-stored accurate responses
- **Context-Aware Generation**: For complex queries, provides relevant document context to the LLM
- **Fallback Responses**: When no relevant context is found, generates general insurance knowledge responses
- **Caching**: Implements intelligent caching to improve response times and reduce API costs

### 4. API Endpoints

The system provides several RESTful endpoints:

- `POST /api/v1/hackrx/run`: Main endpoint for processing queries against documents
- `GET /api/v1/documents`: Retrieve document information
- `GET /api/v1/analytics`: Access query analytics and insights

### 5. Performance Features

- **Batch Processing**: Handles multiple queries efficiently
- **Rate Limiting**: Manages API calls to prevent overloading
- **Model Fallback**: Automatically switches between different LLM models if one fails
- **Local/Cloud LLM Support**: Can use either local Ollama models or cloud-based APIs

### Example Usage

```bash
# Process queries against a document
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "path/to/insurance_policy.pdf",
    "questions": [
      "What is the coverage limit for medical expenses?",
      "What are the exclusions in this policy?"
    ]
  }'
```

The system will return structured responses with accurate answers based on the document content and intelligent analysis.
