"""
Author: Janiya Richardson
Date: 02/24/24


This program is a test to converting files diy style

This program will also attempt in asking the user for a file from their computer and then converting it
"""
#Import necessary libraries
import ebooklib
from ebooklib import epub
import docx
import pypdf

import tkinter as tk
from tkinter import filedialog


def ask_for_file():

    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename()
    if file_path:
       if 'pdf' in file_path:
           def pdf_to_txt(pdf_file):
               text = ''
               with open(pdf_file, 'rb') as file:
                   reader = pypdf.PdfReader(file)
                   for page_num in range(len(reader.pages)):
                       page = reader.pages[page_num]
                       text += page.extract_text()
               return text
           print(pdf_to_txt(file_path))
       elif 'docx' in file_path:
           def docx_to_txt(docx_file):
               doc = docx.Document(docx_file)
               text = ''
               for para in doc.paragraphs:
                   text += para.text + '\n'
               return text
           print(docx_to_txt(file_path))
       elif 'epub' in file_path:
           def epub_to_txt(epub_file):
               book = epub.read_epub(epub_file)
               text = ''
               for item in book.get_items():
                   if isinstance(item, ebooklib.epub.EpubHtml):
                       text += item.get_body_content().decode('utf-8', 'ignore') + '\n'
               return text
           print(epub_to_txt(file_path))

    else:
        print("No file selected.")

ask_for_file()
