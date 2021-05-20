from bertopic import BERTopic

from sklearn.datasets import fetch_20newsgroups


docs = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))['data']
print(len(docs))
print(docs[1])


topic_model = BERTopic(language='english', calculate_probabilities=True)
topics, _ = topic_model.fit_transform(docs)

topic_freq = topic_model.get_topic_freq()
print('#Classes:', len(topic_freq) - 1)
outliers = topic_freq['Count'][topic_freq['Topic'] == -1]
print('#Unclassified documents:', outliers)
print(topic_freq.head())

topic_model.get_topic(topic_freq['Topic'].iloc[1])
fig = topic_model.visualize_topics()
fig.write_html("pass1.html")

