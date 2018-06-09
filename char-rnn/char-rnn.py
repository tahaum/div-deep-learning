from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, CuDNNLSTM, Embedding, LSTM
from keras.callbacks import ModelCheckpoint
from utils import *

# Variables
filepath = 'Data/advs_of_sherlock_holmes.txt'
seq_length = 100
hidden_units = 256
batch_size = 350
max_epochs = 100
embedding_size = 256
clipvalue = 5
val_split = 0.2
lr = 0.01
dropout_1, dropout_2 = 0.4, 0.4
optimizer = 'rmsprop'
loss_func = 'categorical_crossentropy'
output_activation = 'softmax'
generate_length = 200

# Load data
x, y, n_chars, n_seqs, n_vocab = load_data(filepath, seq_length)
n_training_samples = int((1-val_split)*n_seqs)
n_val_samples = int(val_split*n_seqs)
print('Vocabulary size:', n_vocab)
print('Number of training samples:', n_seqs)

# Define model
model = Sequential()
model.add(Embedding(input_dim=n_vocab, output_dim=embedding_size, input_length=seq_length))
model.add(LSTM(hidden_units, stateful=False, return_sequences=True,
    input_shape=(seq_length, embedding_size)))
model.add(Dropout(dropout_1))
model.add(LSTM(hidden_units, stateful=False,  return_sequences=False))
model.add(Dropout(dropout_2))
model.add(Dense(n_vocab, activation = output_activation))
rmsprop = optimizers.RMSprop(lr=lr, clipvalue=clipvalue)
model.compile(loss = loss_func, optimizer = rmsprop)
print(model.summary())

# Define checkpoints
checkpoint_path = '/output/checkpoint-{epoch:02d}-{loss:.2f}-{val_loss:.2f}.hdf5'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1,
    save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# -----------Training the model-----------
# model.fit(x, y, epochs=max_epochs, shuffle=True, batch_size=batch_size, validation_split=val_split,
#    callbacks=callbacks_list, verbose=0)

# -----------Generating text-----------
# Get weights from trained network
filepath = 'checpoint.hdf5'
model.load_weights(filepath)

sample_text, output_text = generate_text(model, x, n_seqs, n_vocab, generate_length)
print('Sample text:\n',sample_text,'\n____________________\n','Output text:\n',output_text, sep='')