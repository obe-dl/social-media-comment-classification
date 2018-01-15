import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import argparse

from utils import read_data


def build_corpus(texts):
    return [text.split(" ") for text in texts]


def tsne_plot(model):
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(128, 128))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
    plt.clf()

parser = argparse.ArgumentParser(description='Create word2vec model from corpus and visualize embeddings using t-SNE.')
parser.add_argument("first_dataset")
parser.add_argument("second_dataset")

args = parser.parse_args()

texts, labels = read_data(args.first_dataset, args.second_dataset)
corpus = build_corpus(texts)
model = Word2Vec(corpus, size=100, window=50, min_count=300, workers=4)
tsne_plot(model)