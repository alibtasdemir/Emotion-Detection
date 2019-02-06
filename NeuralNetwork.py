from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Bidirectional
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import re

def clean_text(text):
	
	# Remove punct.
	translator = str.maketrans('', '', string.punctuation)
	text = text.translate(translator)

	# to lower
	text = text.lower().split()

	# Remove stopwords
	stops = set(stopwords.words('english'))
	text = [w for w in text if not w in stops and len(w) >= 3]

	text = " ".join(text)
	### Cleaning
	text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text) # Numbers and symbols deleted
	
	# Shortened writings fixed.
	text = re.sub(r"what's", "what is ", text)
	text = re.sub(r"\'s", " ", text)
	text = re.sub(r"\'ve", " have ", text)
	text = re.sub(r"n't", " not ", text)
	text = re.sub(r"i'm", "i am ", text)
	text = re.sub(r"\'re", " are ", text)
	text = re.sub(r"\'d", " would ", text)
	text = re.sub(r"\'ll", " will ", text)

	### Stemming
	text = text.split()
	"""
	stemmer = SnowballStemmer('english')
	"""
	stemmer = WordNetLemmatizer()
	stemmed = [stemmer.lemmatize(word) for word in text]
	#stemmed = [stemmer.stem(word) for word in text]
	text = " ".join(stemmed)
	return text

path_emotion_file = "isear.csv"

# read csv
df_file = pd.read_csv(path_emotion_file, error_bad_lines=False,
                      warn_bad_lines=False, sep='|', encoding='latin1')

df = df_file[['Field1', 'SIT']]
old = (df.shape[0])
df = df[~df['SIT'].str.contains('\[')]
df = df.reset_index(drop=True)
print("%d sentences deleted."%(old-df.shape[0]))

df['SIT'] = df['SIT'].map(lambda x: clean_text(x))

classnum=sorted(set(df['Field1']))
macro_to_id = dict((note, number) for number, note in enumerate(classnum))

def fun(i):
    return macro_to_id[i]

df['Field1']=df['Field1'].apply(fun)

labels = []

for idx in df['Field1']:
    labels.append(idx)


vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(df['SIT'])
sequences = tokenizer.texts_to_sequences(df['SIT'])
data = pad_sequences(sequences, maxlen=100)

labels = to_categorical(np.asarray(labels))
print('Shape of Data Tensor:', data.shape) # num of examples x Features
print('Shape of Label Tensor:', labels.shape) # num of examples x Class

# Shuffling
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

nb_test_samples = int(0.2 * data.shape[0])

x_train = data[:-nb_test_samples]
y_train = labels[:-nb_test_samples]
x_test = data[-nb_test_samples:]
y_test = labels[-nb_test_samples:]


def getGlove(filename, vocabulary_size, vectorDim, tokenizer):
	embeddings_index = {}
	f = open(filename,encoding='utf8')
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
	f.close()
	print('Total %s word vectors in %s.' % (len(embeddings_index), filename))
	count = 0
	total = len(tokenizer.word_index.items())
	embedding_matrix = np.zeros((vocabulary_size+1, vectorDim))
	for word, index in tokenizer.word_index.items():
	    if index > vocabulary_size - 1:
	        break
	    else:
	        embedding_vector = embeddings_index.get(word)
	        if embedding_vector is not None:
	            embedding_matrix[index] = embedding_vector
	            count += 1

	return embedding_matrix, total, count


# Get Glove matrix
embedding_matrix, total, count = getGlove("glove.6B.100d.txt", vocabulary_size, 100, tokenizer)
print("Total: %d, Found: %d"%(total, count))

"""
Network 1:
	model.add(Dropout(0.2))
	model.add(Conv1D(64, 5, activation='relu'))
	model.add(MaxPooling1D(pool_size=4))
	model.add(LSTM(100))
	model.add(Dense(len(classnum), activation='softmax'))

Network 2:
	model.add(Conv1D(128, 5, activation='relu'))					#1x96x128
	model.add(MaxPooling1D(3))                               		#1x32x128
	model.add(Conv1D(128, 5, activation='relu'))	                #1x28x128
	model.add(MaxPooling1D(28))		    # global max pooling        #1x1x128
	model.add(LSTM(128))
	model.add(Dense(128, activation='relu'))	                    #1x128
	model.add(Dense(len(classnum), activation='softmax'))			#1x7

Network 3:
	model.add(Conv1D(128, 5, activation='relu'))					#1x96x128
	model.add(MaxPooling1D(3))                               		#1x32x128
	model.add(Conv1D(128, 5, activation='relu'))	                #1x28x128
	model.add(MaxPooling1D(28))		    # global max pooling        #1x1x128
	model.add(Flatten())	                                        #1x128
	model.add(Dense(128, activation='relu'))	                    #1x128
	model.add(Dense(len(classnum), activation='softmax'))	        #1x7

Network 4:
	model.add(Bidirectional(LSTM(100)))
	model.add(Dense(len(classnum), activation='softmax'))

"""
def neuralNet():
	model = Sequential()
	# Vectorizing sequence
	model.add(Embedding(vocabulary_size+1, 100, input_length=100, weights=[embedding_matrix], trainable=True))
	model.add(Bidirectional(LSTM(100)))
	model.add(Dense(len(classnum), activation='softmax'))

	# Compile model and print info screen.
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print("Neural network")
	model.summary()
	return model

def plotModel(model, filename):
	prestr = 'loss_test_'
	lossfile = prestr + filename
	lossfile = lossfile + '.png'

	fig1 = plt.figure()
	plt.plot(model.history['loss'],'r',linewidth=3.0)
	plt.plot(model.history['val_loss'],'b',linewidth=3.0)
	plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
	plt.xlabel('Epochs ',fontsize=16)
	plt.ylabel('Loss',fontsize=16)
	plt.title('Loss Curves :Network 6',fontsize=16)
	fig1.savefig(lossfile)

	prestr = 'accuracy_test_'
	accfile = prestr + filename
	accfile = accfile + '.png'
	fig2=plt.figure()
	plt.plot(model.history['acc'],'r',linewidth=3.0)
	plt.plot(model.history['val_acc'],'b',linewidth=3.0)
	plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
	plt.xlabel('Epochs ',fontsize=16)
	plt.ylabel('Accuracy',fontsize=16)
	plt.title('Accuracy Curves : Network 6',fontsize=16)
	fig2.savefig(accfile)


cb100d=ModelCheckpoint('model_100d.hdf5',monitor='val_acc',verbose=1,save_best_only=True)

model = neuralNet()

#history = model.fit(x_train, np.array(y_train), validation_split=0.2, epochs=10, callbacks=[cb100d])
#plotModel(history, "network6")
model.load_weights("model_100d.hdf5")

def toWord(tokenizer, sequence):
	index_word = {v: k for k, v in tokenizer.word_index.items()}
	words = []
	for seq in sequence:
		if seq != 0:
			words.append(index_word.get(seq))
	return " ".join(words)



def makePredictions(model, texts, labels, tokenizer):
	prediction = model.predict(texts)
	predicted = []
	right = [np.argmax(x) for x in labels]
	for pred in prediction:
		label = np.argmax(pred)
		predicted.append(label)

	accuracy = np.mean(np.array(right) == np.array(predicted))
	print("%d sentences tested." % len(texts))
	print("Accuracy: %.3f"% accuracy)
	return np.array(predicted), np.array(right)

predicted, correct = makePredictions(model, x_test, y_test, tokenizer)

import itertools
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if normalize:
    	plt.savefig("normalized_confusion_network6.png")
    else:
    	plt.savefig("confusion_network6.png")


cnf_matrix = confusion_matrix(correct, predicted)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
"""
plot_confusion_matrix(cnf_matrix, classes=classnum,
                      title='Confusion matrix, without normalization')

plot_confusion_matrix(cnf_matrix, classes=classnum, normalize=True,
                      title='Normalized confusion matrix')
"""
results = model.evaluate(x_test, y_test)
print(results)