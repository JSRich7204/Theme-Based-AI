"""
Author: Janiya Richardson
Date: 03/25/24

This program will generate a mind map based on themes 
between words using a matrix of important words

"""

#Import necessary libraries
import docx
import pypdf

import tkinter as tk
from tkinter import filedialog
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import networkx as nx

# Download NLTK resources 
nltk.download('punkt')
nltk.download('stopwords')

# Get the list of English stopwords
stop_words = stopwords.words('english')

# Add your custom words
stop_words.extend(['really', 'like'])


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

# Define a function to preprocess text
def preprocess_text(text):
    # Convert to lowercase and tokenize
    tokens = word_tokenize(text.lower())
    # Remove stop words and non-alphabetic tokens
    stop_words_set = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words_set and word.isalpha()]
    return tokens

# Define a function to create a TF-IDF matrix
def create_tfidf_matrix(texts):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    return tfidf_matrix, tfidf_vectorizer.get_feature_names_out()

# Define a function to perform topic modeling with NMF
def perform_nmf(tfidf_matrix, n_components=5):
    nmf_model = NMF(n_components=n_components)
    W = nmf_model.fit_transform(tfidf_matrix)
    H = nmf_model.components_
    return W, H

# Define a function to create a mindmap from NMF topics
def create_mindmap(H, feature_names):
    G = nx.Graph()
    for topic_idx, topic in enumerate(H):
        topic_label = f"Topic {topic_idx+1}"
        G.add_node(topic_label)
        for i in topic.argsort()[-10:]:
            G.add_node(feature_names[i])
            G.add_edge(topic_label, feature_names[i])
    return G

# Define a function to visualize the mindmap
def visualize_mindmap(G):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10)
    plt.title("Mindmap of Themes")
    plt.show()

# Preprocess the text
processed_text = preprocess_text(text)

# Create a TF-IDF matrix
tfidf_matrix, feature_names = create_tfidf_matrix([' '.join(processed_text)])

# Perform topic modeling with NMF
W, H = perform_nmf(tfidf_matrix)

# Create a mindmap from NMF topics
mindmap = create_mindmap(H, feature_names)

# Visualize the mindmap
visualize_mindmap(mindmap)