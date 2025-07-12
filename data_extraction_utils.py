# Utility functions for data extraction from Resumes
import fitz # PyMuPDF
import re

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
    variable_name = "履歴書_能登屋　亮"  
    pdf_path = f"Data/{variable_name}.pdf"
    extracted_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(extracted_text)
    print(extracted_text)
    print("=================================================")
    print(cleaned_text)
    print("=================================================")