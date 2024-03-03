"""
Author: Janiya Richardson
Date: 03/03/24


This program will generate a mind map based themes between words by using the pre-trained model GPT
Attempt #3 using clustering
"""

# Import necessary libraries
import docx
import pypdf

import tkinter as tk
from tkinter import filedialog

import torch
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import threadpoolctl

# Use threadpoolctl to limit OpenMP threads
threadpoolctl.threadpool_limits(1)


# Step 1: Convert the file to txt
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

# Load pre-trained GPT2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")

# Tokenize text
tokens = tokenizer.tokenize(text)

# Generate embeddings using GPT2
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
with torch.no_grad():
    outputs = model(**inputs)
embeddings = outputs.last_hidden_state

# Reshape embeddings to 2D
embeddings_2d = embeddings.squeeze(0).numpy()

# Perform clustering
num_clusters = 1  # Example number of clusters
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(embeddings_2d)

# Get cluster labels
cluster_labels = kmeans.labels_

# Identify themes based on cluster centroids
cluster_centers = kmeans.cluster_centers_

# Find most representative tokens for each cluster
most_representative_tokens = []
cluster_embeddings = embeddings_2d[cluster_labels == 0]  # Assuming only one cluster
cluster_similarity = cosine_similarity(cluster_embeddings, [cluster_centers[0]])
most_similar_index = cluster_similarity.argmax()
most_representative_token = tokens[most_similar_index]
most_representative_tokens.append(most_representative_token)

# Construct graph for mind map
G = nx.Graph()

# Add nodes for each theme
for token in most_representative_tokens:
    G.add_node(token)

# Add edges based on relationships between themes using co-occurrence
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

# Adding edges based on co-occurrence relations between themes
for (theme1, theme2), weight in co_occurrences.items():
    theme1_index = tokens.index(theme1)
    theme2_index = tokens.index(theme2)
    G.add_edge(theme1, theme2, weight=weight)

# Visualize the mind map
pos = nx.spring_layout(G)  # Layout algorithm
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10)
plt.show()