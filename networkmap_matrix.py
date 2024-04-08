"""
Author: Janiya Richardson
Date: 3/24/2024

This program will generate a network map from by reading the text line by line and creating a similarity matrix
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

# Define a list of stop words
stop_words = {'the', 'and', 'but', 'however', 'a', 'an', 'in', 'on', 'at', 'for', 'with', 'without', 'under', 'over', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'that', 'Ah', 'Oh'}

def create_word_matrix(text):
    # Convert text to lower case to ignore case sensitivity
    text = text.lower()
    # Split the text into lines
    lines = text.split('\\n')
    # Create a matrix to hold words from each line
    word_matrix = []
    for line in lines:
        # Split the line into words and filter out stop words
        words = [word for word in line.split() if word not in stop_words]
        word_matrix.append(words)
    return word_matrix
    return word_matrix

def find_similar_words(word_matrix):
    # Create a dictionary to hold word counts
    word_count = {}
    # Iterate over each row in the matrix
    for row in word_matrix:
        # Iterate over each word in the row
        for word in row:
            # If the word is already in the dictionary, increment its count
            word_count[word] = word_count.get(word, 0) + 1
    # Find words that appear more than once
    similar_words = {word for word, count in word_count.items() if count > 1}
    return similar_words

def calculate_total_word_pairs(text):
    # Split the text into words
    words = text.split()
    
    # Calculate the total number of word pairs
    total_word_pairs = len(words) - 1 if len(words) > 1 else 0
    
    return total_word_pairs

def calculate_unique_word_pairs(text):
    # Split the text into words
    words = text.lower().split()
    
    # Create a set to store unique word pairs
    unique_word_pairs = set()
    
    # Iterate over the words to form pairs
    for i in range(len(words) - 1):
        # Form a word pair with the next word
        pair = (words[i], words[i + 1])
        # Add the pair to the set
        unique_word_pairs.add(pair)
    
    return len(unique_word_pairs)


def create_network_map(word_matrix, similar_words, text):
    #If the text is too long, only show the unique words in the network map
    if len(text) > 1000:
        # Create a new graph
        G = nx.Graph()
        # Add nodes for each unique word
        for row in word_matrix:
            for word in row:
                if word in similar_words:
                    G.add_node(word)
        # Add edges between similar words with a threshold
        threshold = (calculate_total_word_pairs(text) / calculate_unique_word_pairs(text)) * 1000 # Only add an edge if words appear together more than 'threshold' times
        word_pair_count = {}
        for i, row in enumerate(word_matrix):
            for word1 in row:
                for word2 in row:
                    if word1 in similar_words and word2 in similar_words and word1 != word2:
                        word_pair = tuple(sorted([word1, word2]))
                        word_pair_count[word_pair] = word_pair_count.get(word_pair, 0) + 1
                        if word_pair_count[word_pair] > threshold:
                            G.add_edge(word1, word2)
        return G
    else:
        # Create a new graph
        G = nx.Graph()
        # Add nodes for each unique word
        for row in word_matrix:
            for word in row:
                if word in similar_words:
                    G.add_node(word)
        # Add edges between similar words
        for i, row in enumerate(word_matrix):
            for word1 in row:
                for word2 in row:
                    if word1 in similar_words and word2 in similar_words and word1 != word2:
                        G.add_edge(word1, word2)
        return G  


# Create the word matrix
word_matrix = create_word_matrix(text)

# Find similar words
similar_words = find_similar_words(word_matrix)

# Create the network map
G = create_network_map(word_matrix, similar_words, text)

# Draw the network map
nx.draw(G, with_labels=True)
plt.show()