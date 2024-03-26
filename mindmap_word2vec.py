"""
Author: Janiya Richardson
Date: 03/12/24

This program will generate a mind map based on themes 
between words using the pre-trained model GPT

Attempt #4 using word2vec and clustering
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
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from gensim.models import Word2Vec
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

# Preprocess text
def preprocess(text):
    text = text.lower()
    text = ''.join([word for word in text if word not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Train Word2Vec model
model = Word2Vec([word_tokenize(text)], min_count=1, vector_size=100, window=5, sg=1)

# Get word vectors
word_vectors = model.wv

# Perform KMeans clustering
num_clusters = 5  # Example number of clusters
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(word_vectors.vectors)

# Get cluster labels
word_cluster_labels = kmeans.labels_

# Create a graph for visualization
G = nx.Graph()

# Add nodes and edges based on clustered words
clustered_words = {}
for word, cluster_label in zip(word_vectors.index_to_key, word_cluster_labels):
    clustered_words.setdefault(cluster_label, []).append(word)
    G.add_node(word)
    G.add_edge(f"Cluster {cluster_label}", word)

# Calculate the average length of the node labels
avg_label_length = sum(len(word) for word in G.nodes) / len(G.nodes)

# Adjust the k parameter based on the average label length
k = avg_label_length / 100  # Adjust the denominator as needed
    
# Visualize the graph
pos = nx.spring_layout(G, k)
nx.draw(G, pos, node_color='lightblue', edge_color='gray')

# Calculate the size of the text based on the length of the word
label_sizes = {word: len(word) for word in G.nodes}

# Draw labels with specific size
for node, (x, y) in pos.items():
    plt.text(x, y, node, fontsize=label_sizes[node])

plt.show()