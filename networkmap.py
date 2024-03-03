"""
Author: Janiya Richardson
Date: 03/03/24


This program will generate a network map based similarity and relation between words by using the pre-trained model GPT
Attempt #1
"""

#Import necessary libraries
import docx
import pypdf

import tkinter as tk
from tkinter import filedialog
import torch
import nltk
import networkx as nx
import matplotlib.pyplot as plt
from torch import threshold
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config


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


# Step 1: Text Processing
def preprocess_text(text):
    # Tokenize text into sentences
    sentences = text.split(".")  # Example: split by period for sentences
    return sentences


# Step 2: Identifying Relationships
def build_similarity_matrix(sentences, gpt_model, tokenizer):
    similarity_matrix = torch.zeros(len(sentences), len(sentences))
    for i, sent1 in enumerate(sentences):
        for j, sent2 in enumerate(sentences):
            # Convert sentences to tensors
            inputs1 = tokenizer.encode(sent1, return_tensors="pt", add_special_tokens=True)
            inputs2 = tokenizer.encode(sent2, return_tensors="pt", add_special_tokens=True)
            # Generate contextual embeddings
            outputs1 = gpt_model(inputs1)
            outputs2 = gpt_model(inputs2)
            # Compute cosine similarity
            similarity_matrix[i, j] = torch.cosine_similarity(
                outputs1.last_hidden_state.mean(dim=1), outputs2.last_hidden_state.mean(dim=1), dim=1
            ).item()  # Get the item (float) instead of keeping it as tensor
    return similarity_matrix


# Steps 3 & 4: Constructing the Network & Visualizing the Network
def visualize_network(sentences, similarity_matrix, threshold):
    G = nx.Graph()
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            weight = similarity_matrix[i, j]
            if weight > threshold:
                G.add_edge(sentences[i], sentences[j], weight=weight)

    pos = nx.spring_layout(G)  # or any other layout algorithm
    nx.draw(G, pos, with_labels=True, node_size=50, font_size=10)
    plt.show()


# Example usage
sentences = preprocess_text(text)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
gpt_model = GPT2Model.from_pretrained("gpt2")
similarity_matrix = build_similarity_matrix(sentences, gpt_model, tokenizer)
visualize_network(sentences, similarity_matrix, threshold=0.5)  # Adjust threshold as needed