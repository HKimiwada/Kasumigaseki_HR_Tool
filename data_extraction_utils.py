# Utility functions for data extraction from Resumes
import fitz # PyMuPDF

doc = fitz.open("Data/谷口博城_職務経歴書.pdf")
text = ""
for page in doc:
    text += page.get_text()
print(text)