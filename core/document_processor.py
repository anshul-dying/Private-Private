import PyPDF2
import docx
from loguru import logger
import requests
import os
from urllib.parse import urlparse, parse_qs

class DocumentProcessor:
    def __init__(self):
        self.temp_dir = "temp"
        os.makedirs(self.temp_dir, exist_ok=True)

    def extract_text(self, doc_url: str) -> str:
        logger.info(f"Processing document: {doc_url}")
        # Extract file name from URL, ignoring query parameters
        parsed_url = urlparse(doc_url)
        file_name = os.path.basename(parsed_url.path)
        logger.info(f"Parsed file name: {file_name}")

        if file_name.lower().endswith(".pdf"):
            return self._extract_pdf(doc_url)
        elif file_name.lower().endswith(".docx"):
            return self._extract_docx(doc_url)
        else:
            logger.error(f"Unsupported document type for file: {file_name}")
            raise ValueError(f"Unsupported document type: {file_name}")

    def _extract_pdf(self, doc_url: str) -> str:
        temp_file = os.path.join(self.temp_dir, "temp.pdf")
        response = requests.get(doc_url)
        response.raise_for_status()
        with open(temp_file, "wb") as f:
            f.write(response.content)
        try:
            with open(temp_file, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "".join(page.extract_text() or "" for page in reader.pages)
            logger.info(f"Extracted text from PDF: {len(text)} characters")
            return text
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _extract_docx(self, doc_url: str) -> str:
        temp_file = os.path.join(self.temp_dir, "temp.docx")
        response = requests.get(doc_url)
        response.raise_for_status()
        with open(temp_file, "wb") as f:
            f.write(response.content)
        try:
            doc = docx.Document(temp_file)
            text = "\n".join(para.text for para in doc.paragraphs)
            logger.info(f"Extracted text from DOCX: {len(text)} characters")
            return text
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)