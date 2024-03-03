"""
Author: Janiya Richardson
Date: 03/03/24


This program will generate a mind map using theme extraction by using the pre-trained model BERT
Attempt #1 using BERT
"""

#Import necessary libraries
import docx
import pypdf

import tkinter as tk
from tkinter import filedialog
import torch
import transformers
from nltk.tokenize import word_tokenize
import networkx as nx
import matplotlib.pyplot as plt

#Convert the file to txt
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

txt = ask_for_file()

def preprocess_text(text):
    # Tokenize text into sentences
    sentences = text.split(".")  # Example: split by period for sentences
    return sentences

if txt:
    # Load pre-trained model
    model_name = 'bert-base-uncased'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModel.from_pretrained(model_name)

    # Tokenize the text
    tokens = word_tokenize(preprocess_text(txt))

    # Truncate or split the tokens if they exceed the maximum sequence length
    max_seq_length = 512
    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length]  # Truncate the tokens
        print("Warning: Input sequence exceeds maximum length. Truncated to {} tokens.".format(max_seq_length))

    # Convert tokens to input IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension

    # Extract themes using BERT
    with torch.no_grad():
        outputs = model(input_ids)

    themes = outputs.last_hidden_state.mean(dim=1).squeeze()

    # Co-occurrence relation extraction
    co_occurrences = {}
    window_size = 5  # Define the window size for co-occurrence

    for i in range(len(tokens)):
        for j in range(i + 1, min(i + window_size, len(tokens))):
            theme1 = tokens[i].lower()  # Convert to lowercase
            theme2 = tokens[j].lower()  # Convert to lowercase
            if theme1 != theme2:
                if theme1 in tokens and theme2 in tokens:  # Check if theme tokens exist in tokens list
                    theme1_index = tokens.index(theme1)
                    theme2_index = tokens.index(theme2)
                    key = (theme1, theme2)
                    co_occurrences[key] = co_occurrences.get(key, 0) + 1
                else:
                    continue

    # Generate mind map
    G = nx.Graph()

    # Adding nodes representing themes
    for i, theme in enumerate(themes):
        if i < len(tokens):
            G.add_node(i, label=tokens[i])
        else:
            continue

    # Adding edges based on co-occurrence relations between themes
    for (theme1, theme2), weight in co_occurrences.items():
        theme1_index = tokens.index(theme1)
        theme2_index = tokens.index(theme2)
        G.add_edge(theme1_index, theme2_index, weight=weight)

    # Visualize the mind map
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=False)  # Don't draw default labels
    labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='black', font_family='sans-serif')
    plt.show()