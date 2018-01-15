import numpy as np

from keras.callbacks import TensorBoard
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.utils import plot_model

from utils import fmeasure, precision, recall, plot_accuracy, plot_loss, read_data
import argparse

parser = argparse.ArgumentParser(description='CNN with pre-trained word embeddings.')
parser.add_argument("first_dataset")
parser.add_argument("second_dataset")
parser.add_argument("vector_path")
parser.add_argument("vector_dimension", type=int)

parser.add_argument("--max_word_length")
parser.add_argument("--max_nb_words")
parser.add_argument("--plot")

args = parser.parse_args()

MAX_SEQUENCE_LENGTH = args.max_word_length if args.max_word_length != None else 300
MAX_NB_WORDS = args.max_nb_words if args.max_nb_words != None else 30000
EMBEDDING_DIM = args.vector_dimension
VALIDATION_SPLIT = 0.2

# Read word vectors and create word to embedding vector mapping.
print("Indexing word vectors.")
embeddings_index = {}

f = open(args.vector_path)
for line in f:
    try:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
    except:
        continue
    embeddings_index[word] = coefs
f.close()

print("Total number of word vectors: {}".format(len(embeddings_index)))

texts, labels = read_data(args.first_dataset, args.second_dataset)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print("Shape of data tensor:", data.shape)

x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2)

num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
print("Preparing embedding matrix.")

# Prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print("Build model...")

model = Sequential()
model.add(Embedding(num_words,
                    EMBEDDING_DIM,
                    weights=[embedding_matrix],
                    input_length=MAX_SEQUENCE_LENGTH,
                    trainable=False))
model.add(Conv1D(32, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(32, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(32, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc', fmeasure, precision, recall])
plot_model(model, to_file='model.png', show_shapes=True)

print("Train...")
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=50,
                    validation_data=(x_val, y_val), callbacks=[tensorBoardCallback])


if args.plot:
    plot_accuracy(history.history)
    plot_loss(history.history)
