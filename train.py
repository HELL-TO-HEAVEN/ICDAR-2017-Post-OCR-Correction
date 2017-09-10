import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import sys
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import h5py


# load ascii text and covert to lowercase of input
filename_input = "./data/EngMonoInput.txt"
raw_text1 = open(filename_input).read()
raw_text1 = raw_text1.lower()
# create mapping of unique chars to integers, and a reverse mapping
chars_input = sorted(list(set(raw_text1)))
# load ascii text and covert to lowercase of gs
filename = "./data/EngMonoGS.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers, and a reverse mapping
chars_gs = sorted(list(set(raw_text)))
temp= set(chars_input)-set(chars_gs)
chars = sorted(list(set(chars_gs+ list(temp))))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

#choose a specifix GPU to run session 0 for the default
with tf.device('/gpu:0'):
    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement = True)
    # set % memory will be use to run a session
    config.gpu_options.per_process_gpu_memory_fraction = 0.65 # test for 50% memory size.
    set_session(tf.Session(config=config))
    # define the LSTM model 3 layers
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    #model.add(LSTM(256))
    #model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # define the checkpoint
    filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # fit the model
    model.fit(X, y, epochs=50, batch_size=64, callbacks=callbacks_list)


