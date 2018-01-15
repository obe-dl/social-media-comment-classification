from matplotlib import pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

import json
import numpy as np
import random
import sys
import time

from utils import fmeasure, precision, recall, get_activations


# python classifier.py visualize tr ../trained_models/model-20171209-193942.h5 20 'ayakkabi'
# python classifier.py train tr
# python classifier.py test tr ../trained_models/model-20171209-193942.h5 10

datasets = {
    'tr': (
        '../datasets/akp-chp/akp.txt',
        '../datasets/akp-chp/chp.txt'
    ),
    'en': (
        '../datasets/trump-clinton/trump.txt',
        '../datasets/trump-clinton/clinton.txt'
    )
}

if len(sys.argv) < 2:
    print('Enter arguments')
    sys.exit(0)

dataset = datasets[sys.argv[2]]
batch_size = 1024
sequence_length = 40
sequence_count = int(1e+05)
epochs = 70
step = None


def preprocess(text):
    if step:
        # sliding window on comments in order to utilize sentences more
        windowed_text = []
        for txt in text:
            for i in range(0, len(txt) - sequence_length, step):
                windowed_text.append(txt[i: i + sequence_length])

        print('Windowed sequences 0: {}'.format(len(windowed_text)))
        text = windowed_text

    random.shuffle(text)

    if len(text) < sequence_count:
        print('Not enough sequences')
        sys.exit(0)

    # we can't train all of them if data is too big
    return text[:sequence_count]


text0 = open(dataset[0]).read().lower().split('\n')
text1 = open(dataset[1]).read().lower().split('\n')
print('Text0 sequences: {} Text1 sequences: {}'.format(len(text0), len(text1)))

text0 = preprocess(text0)
text1 = preprocess(text1)

# first set is labeled with 0, second with 1
text0_labels = np.zeros(len(text0), dtype=np.bool)
text1_labels = np.ones(len(text1), dtype=np.bool)

sentences = text0 + text1
print('total sentence count {}'.format(len(sentences)))

# trim the text if it's longer
sentences = list(map(lambda s: s[:sequence_length], sentences))
Y = np.concatenate((text0_labels, text1_labels)).reshape((-1, 1))

chars = ['\n', ' ', '.', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
         'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

X = np.zeros((len(sentences), sequence_length, len(chars)), dtype=np.bool)
for i, s in enumerate(sentences):
    for t, char in enumerate(s):
        X[i, t, char_indices[char]] = 1

if sys.argv[1] == 'train':
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=True)

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    model.add(LSTM(128, input_shape=(sequence_length, len(chars)), return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy', fmeasure, precision, recall])

    print('Train...')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test))
    model_identifier = '{}'.format(time.strftime("%Y%m%d-%H%M%S"))
    model.save('../trained_models/model-{}.h5'.format(model_identifier))
    score, acc, _, _, _ = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    train_accs = history.history['acc']
    test_accs = history.history['val_acc']
    train_losses = history.history['loss']
    test_losses = history.history['val_loss']
    plt.subplot(2, 1, 1)
    plt.plot(range(epochs), train_accs, label="Train Accuracy")
    plt.plot(range(epochs), test_accs, label="Test Accuracy")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(range(epochs), train_losses, label="Train Loss")
    plt.plot(range(epochs), test_losses, label="Test Loss")
    plt.legend()
    plt.savefig('../figures/figure-{}.png'.format(model_identifier))


elif sys.argv[1] == 'test':
    iters = int(sys.argv[4])
    model_path = sys.argv[3]
    model = load_model(model_path)
    mask = np.random.randint(0, 2 * sequence_count - 1, size=(iters,))
    chosen_x = X[mask]
    chosen_y = Y[mask]
    chosen_s = [sentences[i] for i in mask]

    preds = model.predict_classes(chosen_x, batch_size=batch_size, verbose=0)
    human_preds = np.zeros(preds.shape)
    for i in range(iters):
        cur_s = chosen_s[i]
        cur_x = chosen_x[i]
        cur_y = chosen_y[i]
        cur_pred = preds[i]
        human_pred = int(input('AKP->0 CHP->1\n{}\n'.format(cur_s)))
        human_preds[i, 0] = human_pred
        print('Ground truth: {}\nYour prediction: {}\nModel prediction: {}\n'
              .format(int(cur_y), human_pred, cur_pred[0]))
    print('Your accuracy: {}'.format(np.sum(human_preds == chosen_y) / preds.shape[0]))
    print('Model accuracy: {}'.format(np.sum(preds == chosen_y) / preds.shape[0]))


elif sys.argv[1] == 'visualize':
    iters = int(sys.argv[4])
    model_path = sys.argv[3]
    model = load_model(model_path)

    try:
        substr = sys.argv[5]
    except IndexError:
        substr = None

    if substr:
        sentence_idxs = [i for i in range(len(sentences)) if substr in sentences[i]]
        if len(sentence_idxs) < iters:
            print('Not enough sentences with word {}'.format(substr))
            sys.exit(0)

        _mask = np.random.randint(0, len(sentence_idxs) - 1, size=(iters,))
        mask = np.array([sentence_idxs[idx] for idx in _mask])
    else:
        mask = np.random.randint(0, 2 * sequence_length - 1, size=(iters,))

    chosen_x = X[mask]
    chosen_y = Y[mask]
    chosen_s = [sentences[i] for i in mask]

    activations = get_activations(model, 0, chosen_x)

    real_data = []
    for i in range(iters):
        s = chosen_s[i]
        datum = {'pca': [], 'seq': s}
        for j in range(len(s)):
            datum['pca'].append(list(map(lambda x: float(x), activations[i, j, :])))
        real_data.append(datum)

    datasets = {'data': real_data}
    with open('cell.json', 'w') as outfile:
        json.dump(datasets, outfile)

    print('Wrote to json')
