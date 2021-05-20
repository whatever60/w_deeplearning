from sentence_transformers import SentenceTransformer

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

import umap
import hdbscan


def CTfIDF(documents):
        # num_clusters x num_words
        count = CountVectorizer(stop_words='english').fit(documents)
        X = count.transform(documents).toarray()
        tf = X / (X.sum(axis=1) + 1).reshape(-1, 1)  # num_clusters x num_words
        idf = np.log(len(documents) / ((X > 0).sum(axis=0) + 1)) # num_words
        return tf * idf, count


if __name__ == '__main__':
    # Loading data and BERT
    docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    embeddings = model.encode(docs)
    print(embeddings.shape)

    # UMAP
    umap_embeddings = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric='cosine'
    ).fit_transform(embeddings)
    print(umap_embeddings.shape)

    # HDBSCAN
    cluster = hdbscan.HDBSCAN(
        min_cluster_size=10,
        metric='euclidean',
        cluster_selection_method='eom',
    ).fit(umap_embeddings)
    print('#Clusters:', np.unique(cluster.labels_).shape[0] - 1)
    print(pd.Series(cluster.labels_).value_counts())

    # CTF-IDF
    docs_df = pd.DataFrame(docs, columns=['document'])
    docs_df['topic'] = cluster.labels_
    docs_per_topic = docs_df.groupby(['topic']).agg({'document': ' '.join})
    docs_per_topic.reset_index(inplace=True)
    tf_idf, count = CTfIDF(docs_per_topic['document'].values)
    top_word_indices = np.argpartition(tf_idf, -10)[:, -10:]
    words = np.array(count.get_feature_names())
    top_n_words = {
        label: list(zip(words[indices], tf_idf_row[indices].round(6)))
        for label, indices, tf_idf_row in zip(docs_per_topic.topic, top_word_indices, tf_idf)
    }
    print(top_n_words[42])
