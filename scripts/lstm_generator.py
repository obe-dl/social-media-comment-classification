from __future__ import print_function

from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM
from keras.optimizers import RMSprop

import numpy as np
import random
import sys
import time

trump_path = '../datasets/trump-clinton/trump.txt'
clinton_path = '../datasets/trump-clinton/clinton.txt'
akp_path = '../datasets/akp-chp/akp.txt'
chp_path = '../datasets/akp-chp/chp.txt'

batch_size = 1024
sequence_length = 40
sequence_count = int(4e+05)
epochs = 50
step = 10
output_length = 200

text = open(akp_path).read().lower()

text0 = text.split('\n')
print('Text1 sequences: {}'.format(len(text0)))

# sliding window on texts in order to utilize sentences more

windowed_text0 = []
next_chars = []
for t in text0:
    for i in range(0, len(t) - sequence_length, step):
        windowed_text0.append(t[i: i + sequence_length])
        next_chars.append(t[i + sequence_length])

print('Windowed sequences 0: {}'.format(len(windowed_text0)))

random_mask = np.arange(len(windowed_text0))
np.random.shuffle(random_mask)

windowed_text0 = np.array(windowed_text0)[random_mask]
next_chars = np.array(next_chars)[random_mask]

if len(windowed_text0) < sequence_count:
    print('Not enough sequences')
    sys.exit(0)

# we can't train all of them if data is too big
sentences = windowed_text0[:sequence_count]
next_chars = next_chars[:sequence_count]
print('total sentence count {}'.format(len(sentences)))

# trim the text if it's longer(which doesn't happen when we do sliding window)
sentences = list(map(lambda s: s[:sequence_length], sentences))

print('Vectorization...')
chars = ['\n', ' ', '.', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
         'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

X = np.zeros((len(sentences), sequence_length, len(chars)), dtype=np.bool)
Y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, s in enumerate(sentences):
    for t, char in enumerate(s):
        X[i, t, char_indices[char]] = 1
    Y[i, char_indices[next_chars[i]]] = 1

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(sequence_length, len(chars)), return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


generation_identifier = '{}'.format(time.strftime("%Y%m%d-%H%M%S"))

# train the model, output generated text after each iteration
for iteration in range(1, (epochs + 1) // 4):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, Y, batch_size=batch_size, epochs=4)

    while 1:
        sentence_idx = random.randint(0, len(text0))
        if len(text0[sentence_idx]) > sequence_length:
            break

    char_idx = random.randint(0, len(text0[sentence_idx]) - sequence_length - 1)

    with open('../generated/{}.txt'.format(generation_identifier), 'a') as f:
        f.write('iteration: {}\n'.format(iteration))

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text0[sentence_idx][char_idx: char_idx + sequence_length]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(output_length):
            x_pred = np.zeros((1, sequence_length, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

        with open('../generated/{}.txt'.format(generation_identifier), 'a') as f:
            f.write('diversity: {}\n'.format(diversity))
            f.write(generated)
            f.write('\n\n')
