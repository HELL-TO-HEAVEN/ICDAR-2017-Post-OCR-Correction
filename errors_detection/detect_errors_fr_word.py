import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import operator
import os 
import glob
import operator

# load ascii text and covert to lowercase of input
filename_input = "./data/France_Input.txt"
raw_text1 = open(filename_input).read()
raw_text1 = raw_text1.lower()
# create mapping of unique chars to integers, and a reverse mapping
chars_input = sorted(list(set(raw_text1)))
# load ascii text and covert to lowercase of gs
filename = "./data/FranceGS.txt"
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
seq_length = 20
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)


# define the LSTM model
model = Sequential()
# model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
# model.add(Dropout(0.2))
# model.add(Dense(y.shape[1], activation='softmax'))

model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))


# load the network weights
filename = "weights-improvement-10-1.4645-fap.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

path = './ICDAR2017_datasetPostOCR_Evaluation_2M/task1_detection/fr_monograph/'
with open("task_1_result.json","w") as output_file:
    output_file.write("{")
    for filename in glob.glob(path + '*.txt'):
        print filename
        output_file.write("\"" +filename + "\"" +": { \n ") 
        ocr_input=[]
        clean_ocr_input=[]
        with open(filename) as f:
            lines = f.readlines()
            ocr_input.append(lines)

        clean_ocr_input = ocr_input[0][0]
        l = list(clean_ocr_input)
        del(l[0:14])
        sentence = "".join(l)
        sentence= sentence.lower()  
#         print sentence
        chars_data = sorted(list(set(sentence)))
        sequence = sentence.decode('utf-8')
#         print sequence
        for j in range(0,len(sentence)):
            for c in chars_data:
                if c not in chars:
                    sentence = sentence.replace(c, '@')
#         print "Input:" , sentence
        print "\n"
        
#         print 'Sentence:',sentence
        seq_length = 20
        dataX1 = []
        dataY1 = []
        for i in range(0, (len(sentence) - seq_length +1), 1):
            seq_in = sentence[i:i + seq_length]
            dataX1.append([char_to_int[char] for char in seq_in])
        a = (' ', "'", '!', ',', '.','\n','\r')
        b=  ( '_','?','!', "'", ',', '-', '.','0','#', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')
        
         # split the started index
        temp4 = []
        for i in range(0, len(sentence)): 
            if ( sentence[i] == ' '):
                temp4.append(i)
        #decode sequence to utf-8 for handling change of data        
        
        temp2 = []
        for i in range(0, len(sequence)): 
            if ( sequence[i] == ' '.decode('utf-8')):
                temp2.append(i)
        
        temp=[]
        for i in range(0, len(dataX1)-1):
            # print seed
            if (int_to_char[dataX1[i][len(dataX1[i])-1]] != " " ):
                x = numpy.reshape(dataX1[i], (1, len(dataX1[i]), 1))
                x = x / float(n_vocab)
            #         print("List of Probability:")
                prediction = model.predict(x, verbose=0)
            #         print(prediction)
                if ((prediction[0,dataX1[i+1][len(dataX1[i+1])-1]] < 0.0005) and (int_to_char[dataX1[i+1][len(dataX1[i+1])-1]] not in a)):
#                     print "Seed: ",i
#                     print "\"", ''.join([int_to_char[value] for value in dataX1[i]]), "\""
#                     print "Check prob of next character in input: "
#                     print int_to_char[dataX1[i+1][len(dataX1[i+1])-1]], "->", prediction[0,dataX1[i+1][len(dataX1[i+1])-1]]

#                     index = prediction[0].argsort()[::-1][:2]
#                     print "Predicted character and prob:" 
#                     for j in range(0, len(index)):
#                         result = int_to_char[index[j]]
#                         print result , "->", prediction[0,index[j]]
#                     print "\n"
                    temp.append(i)

         #find index of error
        temp3 = []
        for j in range(0, len(temp2)):
            if (temp4[j] != temp2[j]):
                markdown = temp4[j] - temp2[j]    
            for k in range(0, len(temp)):
                if (j< (len(temp2)-1)):
                    if ((temp4[j] <= (temp[k] + 20)) and (temp4[j+1] >= (temp[k] +20))):
        #                 print temp4[j],temp2[j],temp[k], temp[k]+30
                        temp3.append(temp2[j]+1)  
                else: 
                    if (temp4[j] <= (temp[k] + 20)):
        #                 print temp4[j],temp2[j],temp[k], temp[k]+30
                        temp3.append(temp2[j]+1)
        d = {x:temp3.count(x) for x in temp3}
        sorted_x = sorted(d.items(), key=operator.itemgetter(0))
#         print sorted_x
        print "\n"
         # write file
        for i in range(0,len(sorted_x)):
            if (i < len(sorted_x)-1):
                output_file.write( "\"" +str(sorted_x[i][0]) + ":" + str(1) + "\"" +": {} , \n ")
            else:
                output_file.write( "\"" +str(sorted_x[i][0]) + ":" + str(1) + "\"" +": {} \n ")
        output_file.write("} , \n ")
    output_file.write("}")

