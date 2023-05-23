import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import torch
from transformers import BertTokenizer, BertModel

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


def document_level_topic_modeling(documents, num_topics):
    # Load the pre-trained BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Tokenize the documents and obtain the BERT embeddings
    input_ids = []
    attention_masks = []
    for document in documents:
        encoded_dict = tokenizer.encode_plus(
            document,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)

    embeddings = outputs[0][:, 0, :].numpy()  # Use the [CLS] token's embedding as document representation

    # Use Latent Dirichlet Allocation (LDA) for document-level topic modeling
    lda = LatentDirichletAllocation(n_components=num_topics)
    lda.fit(embeddings)

    return lda

# Example corpus for word-level topic modeling
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Example documents for document-level topic modeling
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?"
]

# Set the number of topics and top words for word-level topic modeling
num_topics_word_level = 2
num_top_words_word_level = 3

# Set the number of topics for document-level topic modeling
num_topics_document_level = 2

# Obtain word-level topics
word_level_topics = word_level_topic_modeling(corpus, num_topics_word_level, num_top_words_word_level)
print("Word-level Topics:")
for topic_idx, topic