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

# load ascii text and covert to lowercase
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
filename = "weights-improvement-19-1.5601.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

with open("task_2_result.json","w") as output_file:
    output_file.write("{")
    with open('erroneous_tokens_pos_eng_test.json') as data_file:
        data = json.loads(data_file.read())
        
    for key, value in data.iteritems():
        print key
        output_file.write("\"" +key+ "\"" +": { \n ") 
        ocr_input=[]
        clean_ocr_input=[]
        with open(key) as f:
            lines = f.readlines()
            ocr_input.append(lines)

        clean_ocr_input = ocr_input[0][0]
        l = list(clean_ocr_input)
        del(l[0:14])
        input_OCR = "".join(l)
        input_OCR= input_OCR.lower()  
        chars_data = sorted(list(set(input_OCR)))
        input_decode_utf8 = input_OCR.decode('utf-8')
        for j in range(0,len(input_OCR)):
            for c in chars_data:
                if c not in chars:
                    input_OCR = input_OCR.replace(c, '@')
    #     print 'input_OCR:', input_OCR

        a = (' ', "'", ',', '.','\n','\r')
        b=  ( '_','?','!', "'", ',', '-', '.','0','#', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')
        # split the started index
        temp4 = []
        for i in range(0, len(input_OCR)): 
            if ( input_OCR[i] == ' '):
                temp4.append(i)
        #decode input_decode_utf8 to utf-8 for handling change of data        

        temp2 = []
        for i in range(0, len(input_decode_utf8)): 
            if ( input_decode_utf8[i] == ' '.decode('utf-8')):
                temp2.append(i)
        a = ("'", ',', '.','\n','\r')
        b=  ( '_','?','!', "'", ',', '-', '.','0','#', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')

        check = 0
        for i in range(0, len(value)):
            temp_string =  value[i].encode("utf-8")
            string = "\"" + temp_string+"\"" + ":{"
            temp3 = value[i].split(":")
            check_error= 0
            if (int(temp3[1]) == 1 ):
                for j in range(0, len(temp2)-1):
                    check = 0
                    if ((int(temp3[0])-1) == temp2[j]):
            #             print temp3[0], temp2[j], temp2[j+1], temp4[j], temp4[j+1]
                        sentence =  input_OCR[temp4[j]-20:temp4[j+1]+1]
#                         sentence = sentence1 + "       "
#                         print "Sentence 1 token : " + "\"" + sentence + "\""
                        seq_length = 20
                        dataX1 = []
                        dataY1 = []
                        for num in range(0, (len(sentence) - seq_length +1), 1):
                            seq_in = sentence[num:num + seq_length]
                            dataX1.append([char_to_int[char] for char in seq_in])
                        temp_string1=" "
                        for k in range(1, len(dataX1)-1):
                            # print seed
                        
                            if (int_to_char[dataX1[k][len(dataX1[k])-1]] != " " ):
                                x = numpy.reshape(dataX1[k], (1, len(dataX1[k]), 1))
                                x = x / float(n_vocab)
                                #print("List of Probability:")
                                prediction = model.predict(x, verbose=0)
#                                 print prediction
                                index = prediction[0].argsort()[::-1][:2]
#                                 print index
                                # print(prediction)
                                if (int_to_char[dataX1[k][len(dataX1[k])-1]] in b ): 
                                    if ((prediction[0,dataX1[k+1][len(dataX1[k+1])-1]] < 0.04) and (int_to_char[dataX1[k+1][len(dataX1[k+1])-1]] not in a) and ((prediction[0,index[0]] / prediction[0,dataX1[k+1][len(dataX1[k+1])-1]] ) > 10 ) ):
                                        check_error +=1
                                        check +=1
#                                         print "Seed: ",k
#                                         print "\"", ''.join([int_to_char[value1] for value1 in dataX1[k]]), "\""
#                                         print "Check prob of next character in input: "
#                                         print int_to_char[dataX1[k+1][len(dataX1[k+1])-1]], "->", prediction[0,dataX1[k+1][len(dataX1[k+1])-1]]
#                                         print "Predicted character and prob:" 
                                        if (check >0):
                                            check -=1
                                            up = 0
#                                             if (int_to_char[index[0]] == " "):
#                                                 for m in range(k, len(dataX1)-1):
#                                                     dataX1[m+1].insert(len(dataX1[m+1])-1-up, " ")
#                                                     del dataX1[m+1][-1]
#                                                     dataX1[m+1][len(dataX1[m+1])-1-up] = index[0]
#                                                     up+=1
#                                             else:
                                            for m in range(k, len(dataX1)-1): 
                                                dataX1[m+1][len(dataX1[m+1])-1-up] = index[0]
                                                up+=1

                                        temp_string1=""
                                        for n in range(0, len(index)):
                                            result = int_to_char[index[n]]
                                            # print result , "->", prediction[0,index[n]]
                                            # print sentence[seq_length:k+20]+result+ sentence[k+21:len(sentence)] +  "->", prediction[0,index[n]] 
                                            index_pos = [p for p, q in enumerate(dataX1[k]) if q == 1]
                                            index_pos = max(index_pos)
#                                             print ''.join([int_to_char[value1] for value1 in dataX1[k][index_pos:len(dataX1[k])]]) +result+ sentence[k+21:len(sentence)] +  "->", prediction[0,index[n]] 
                                            if (n < len(index)-1):
                                                temp_string1 = temp_string1+"\"" +''.join([int_to_char[value1] for value1 in dataX1[k][index_pos+1:len(dataX1[k])]])+result+sentence[k+21:len(sentence)-1]+ "\"" +  ":" + str(prediction[0,index[n]]) + ", "             
                                            else:
                                                temp_string1 = temp_string1+"\"" +''.join([int_to_char[value1] for value1 in dataX1[k][index_pos+1:len(dataX1[k])]])+result+sentence[k+21:len(sentence)-1]+ "\"" +  ":" + str(prediction[0,index[n]])         
#                                                
#                             else:
#                                 temp_string1 = ""
#                                 if (n < len(index)-1):
#                                         temp_string1 = temp_string1 + "\"" +input_OCR[temp4[j]:temp4[j+1]+1] + "\"" +  ":" +str(float("{0:.6f}".format(random.uniform(0.9, 1)))) + ", "             
#                                 else:
#                                         temp_string1 = temp_string1 + "\"" +input_OCR[temp4[j]:temp4[j+1]+1] + "\"" +  ":" + str(float("{0:.6f}".format(random.uniform(0.9, 1))))         

                               

                        if (i< len(value)-1):
                            string = string + temp_string1 + "},"
                        else:  
                            string = string + temp_string1 + "}"
            else:
                check_error +=1
                for j in range(0, len(temp2)):
                    if ((int(temp3[0])-1) == temp2[j]):
    #                     sentence = input_OCR[temp4[j]-20:temp4[j+int(temp3[1])]]
                        temp_data =  input_OCR[temp4[j]:temp4[j+int(temp3[1])]]
                        temp_data = temp_data.replace(" ", "")
                        sentence =  input_OCR[temp4[j]-20:temp4[j]+1] + temp_data
    #                     print "Sentence 2 token : " + "\"" + sentence + "\""
    #                     print "Demo output: ", temp_data , "-> ", float("{0:.6f}".format(random.uniform(0.9, 1)))
                        if (i < len(value)-1):
                            string = string + "\""+ temp_data + "\"" + ": " +  str(float("{0:.6f}".format(random.uniform(0.9, 1))))   + "},"
                        else:
                            string = string + "\""+ temp_data + "\"" + ": " +  str(float("{0:.6f}".format(random.uniform(0.9, 1))))   + "}"

            if (check_error == 0):
                if (i< len(value)-1):
                    string = "\"" + temp_string+"\"" + ":{ }, "
                else:  
                    string = "\"" + temp_string+"\"" + ":{ }"
                
#             print "check", check_error    
            output_file.write(string + "\n")
#             print "Output" , string
#             print "\n"
        output_file.write("} , \n ")0
#         print "\n"
    output_file.write("}\n ")

