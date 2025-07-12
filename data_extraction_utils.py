# Utility functions for data extraction from Resumes
import fitz # PyMuPDF

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

if __name__ == "__main__":
    # Example usage
    variable_name = "○設計_小川佳英_ヒューマン【職務経歴書】"  
    pdf_path = f"Data/{variable_name}.pdf"
    extracted_text = extract_text_from_pdf(pdf_path)
    print(extracted_text)