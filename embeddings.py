import numpy as np
import pandas as pd
import gensim
from pathlib import Path
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

downloads_path = str(Path.home() / "Downloads")

data = pd.read_excel(downloads_path + '/data/first_wave.xlsx')

# Preprocess the data (tokenize, remove stop words, etc.)

# Train the word embeddings model
model = Word2Vec(data['tokens'], min_count=1)

# Get the word vectors
word_vectors = model.wv.vectors

# Cluster the word vectors using k-means
kmeans = KMeans(n_clusters=10, random_state=0).fit(word_vectors)

# Get the cluster assignments for each word
cluster_assignments = kmeans.labels_

# Assign topics to each document based on the most common cluster assignments
topics = []
for doc_tokens in data['tokens']:
    topic_counts = np.zeros(10)
    for token in doc_tokens:
        if token in model.wv.vocab:
            cluster = cluster_assignments[model.wv.vocab[token].index]
            topic_counts[cluster] += 1
    topics.append(np.argmax(topic_counts))

# Print the topics for each document
for i, topic in enumerate(topics):
    print('Document {}: Topic {}'.format(i, topic))