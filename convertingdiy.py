"""
Author: Janiya Richardson
Date: 02/24/24


This program is a test for converting files diy style

This program will also attempt in asking the user for a file from their computer and then converting it
"""
#Import necessary libraries
import docx
import pypdf

import tkinter as tk
from tkinter import filedialog
import nltk
import networkx as nx
import matplotlib.pyplot as plt



#Step 1: Convert the file to txt
def ask_for_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename()
    if file_path:
        if 'pdf' in file_path:
            return pdf_to_txt(file_path)
        elif 'docx' in file_path:
            return docx_to_txt(file_path)
        elif 'txt' in file_path:
            with open(file_path, 'r') as file:
                return file.read()
    else:
        print("No file selected.")
        return ""

def pdf_to_txt(pdf_file):
    text = ''
    with open(pdf_file, 'rb') as file:
        reader = pypdf.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def docx_to_txt(docx_file):
    doc = docx.Document(docx_file)
    text = ''
    for para in doc.paragraphs:
        text += para.text + '\n'
    return text



text = ask_for_file()

print(len(text))