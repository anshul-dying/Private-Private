import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_REFERER = os.getenv("OPENROUTER_REFERER", "https://github.com/your-repo")

# Local LLM Configuration
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/api/generate")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3.2:3b")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///database.db")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# File Processing Configuration
TEMP_DIR = os.getenv("TEMP_DIR", "temp")
TEMP_FILE_CLEANUP_RETRIES = int(os.getenv("TEMP_FILE_CLEANUP_RETRIES", "3"))
TEMP_FILE_CLEANUP_DELAY = float(os.getenv("TEMP_FILE_CLEANUP_DELAY", "1.0"))
DOWNLOAD_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", "30"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))  # 100MB default limit

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")
if not OPENROUTER_REFERER:
    raise ValueError("OPENROUTER_REFERER not found in .env file")

print(OPENROUTER_API_KEY)