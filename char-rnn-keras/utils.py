import numpy as np
from keras.utils import np_utils

def load_data(filepath, seq_length):
    global int_to_char, char_to_int

    with open(filepath, 'r', encoding='utf-8') as f:
    	text = f.read()

    text = text.lower()
    chars = sorted(list(set(text)))
    char_to_int = dict((c,i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    n_chars = len(text)
    n_vocab = len(chars)

    dataX = list()
    dataY = list()
    for i in range(0,n_chars - seq_length,1):
        seq_in = text[i:i + seq_length]
        seq_out = text[i+seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])

    n_seqs = len(dataX)
    x = np.reshape(dataX, (n_seqs, seq_length))
    y = np_utils.to_categorical(dataY, num_classes=n_vocab)

    return x, y, n_chars, n_seqs, n_vocab

def generate_text(model, x, n_seqs, n_vocab, output_length):
    np.random.seed()
    sample_batch = np.random.randint(0,n_seqs)
    seq_seed = list(x[sample_batch])
    seq = seq_seed
    ixes = list()
    for i in range(output_length):
        x = np.reshape(seq, (1,100))
        p = model.predict(x)
        # Sampling from the softmax vector:
        ix = np.random.choice(range(n_vocab), p = np.reshape(p,n_vocab))
        # Could be done by taking the max softmax-value:
        # ix = np.argmax(np.reshape(p,n_vocab))
        seq.append(ix)
        seq = seq[1:len(seq)]
        ixes.append(ix)

    sample_text = ''.join(int_to_char[integ] for integ in seq_seed)
    output_text = ''.join(int_to_char[integ] for integ in ixes)

    return sample_text, output_text