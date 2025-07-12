# Utility functions for data extraction from Resumes
from dotenv import load_dotenv
from openai import OpenAI
import fitz # PyMuPDF
import re
import os

# Load environment variables from .env
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text],model=model)
    return response.data[0].embedding

def should_process(filename: str) -> bool:
    """Only process PDFs whose name includes 職務経歴書."""
    return '職務経歴書' in filename

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    Args:
        pdf_path (str): Path to the PDF file.  
    Returns:
        str: Extracted text from the PDF.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_text(text):
    """
    Cleans the extracted text by removing unnecessary characters and formatting.
    Args:
        text (str): The raw text extracted from the PDF.
    Returns:
        str: Cleaned text.
    """
    # Remove multiple spaces and newlines
    cleaned_text = re.sub(r'\s+', ' ', text)
    # Remove leading and trailing spaces
    cleaned_text = cleaned_text.strip()
    return cleaned_text

if __name__ == "__main__":
    # Example usage
    variable_name = "鈴木智也_職務経歴書"  
    pdf_path = f"Data/{variable_name}.pdf"
    extracted_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(extracted_text)
    print(extracted_text)
    print("=================================================")
    print(cleaned_text)
    print("=================================================")