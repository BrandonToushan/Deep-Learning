###Determining if news is "fake" via NLP & a NN (99% Accuracy)

###Dataset: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

#Base package imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from bs4 import BeautifulSoup #for pulling data out of html and xml files
import re, string,unicodedata
from string import punctuation

#Natural language toolkit imports
import nltk 
from nltk.corpus import stopwords #words to be ignored
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer #used produce variants of a root/base word
from nltk.stem import WordNetLemmatizer #used find the root of a word variant (lemma)
from nltk.tokenize import word_tokenize,sent_tokenize #strips strings into tokens
#tokenizing text represents every word with a number, changes all chr to lower case
from nltk.tokenize.toktok import ToktokTokenizer #tokenizes final period only
from nltk import pos_tag #tags words
from nltk.corpus import wordnet #English language database

#Tensorflow & Keras imports
import tensorflow as tf
import keras
from keras.preprocessing import text, sequence
from keras.models import Sequential 
from keras.layers import Dense,Embedding,LSTM,Dropout
from keras.callbacks import ReduceLROnPlateau
#plain stack of layers, each layer has one input and one output tensor (Sequential)
#regular deeply connected nn layer (Dense)
#a dense vector rep for words (Embedding)
#in this case a vector represents the projection of the word into a continuous vector space
#the position of the word in the learned vector space is its embedding
#lstm layer (long short-term memory layer) chooses different implementations
#to maximize performance of the NN
#dropout layer helps prevent overfitting by randomly setting input units to 0 at each step during training
#ReduceLROnPlateau reduces the learning rate when the metric has stopped improving

#Sklearn imports
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split

#Loading in datasets
true = pd.read_csv("True.csv")
false = pd.read_csv("Fake.csv")

#Taking a look at the data

#True data
true.head()

#False data
false.head()

#Visualizing breakdown by subject (Fake News)
plt.figure(figsize=(20,10))
sns.countplot('subject',data=true)
plt.show()
print(true.subject.value_counts())

#Visualizing breakdown by subject (Real News)
plt.figure(figsize=(20,10))
sns.countplot('subject',data=false)
plt.show()
print(false.subject.value_counts())

#Merging the two datasets together 
true['category'] = 1
false['category'] = 0
df = pd.concat([true,false]) 
df.head()

#Visualizing breakdown by subject (Real News)
plt.figure(figsize=(20,10))
sns.countplot('category',data=df)
plt.show()

#Checking for missing values
df.isna().sum()

#Topics in subject col are diff for both categories, thus we exclude
df['text'] = df['text'] + " " + df['title']
del df['title']
del df['subject']
del df['date']

#Setting stopwords (English)
stop_words = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop_words.update(punctuation)

#Cleaning the data

#Defining a function to strip html format text
def html_strip(text):
    soup = BeautifulSoup(text,'html.parser') #using BeautifulSoup parser 
    return soup.get_text()

#Defining a function to strip text between square brackets
def square_bracket_strip(text):
    return re.sub('\[[^]]*\]','',text)

#Defining a function to strip url formatted text
def url_strip(text):
    return re.sub(r'http\S+', '', text)

#Defining a function to identify and remove stopwords
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop_words:
            final_text.append(i.strip())
    return " ".join(final_text)

#Defining a function to remove any noise in the text
def remove_noise(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text)
    return text
df['text'] = df['text'].apply(denoise_text)

#Performing train-validation split
X_train, X_val, y_train, y_val = train_test_split(df.text, df.category, random_state = 7,test_size = 0.30)

#Tokenizing text data

#Defining tokenizer
tokenizer = text.Tokenizer(num_words = max_features)

#Tokenizing X_train
tokenizer.fit_on_texts(X_train)
tokenized_train = tokenizer.texts_to_sequences(X_train)
X_train = sequence.pad_sequences(tokenized_train,maxlen = maxlen)

#Tokenizing X_val
tokenized_test = tokenizer.texts_to_sequences(X_val)
X_val = sequence.pad_sequences(tokenized_test, maxlen=maxlen)

#Using the GloVe Method, ie Global vectors for word representation
#Glove derives semantic relationships between words from the co-occurence matrix
#Need three words a time to measure the semantic similarity between words
#ie P(k|ice), P(k|steam) & P(k|ice)/P(k|steam)

#Using pre-trained twitter data from: https://nlp.stanford.edu/projects/glove/
embedding_file = 'glove.6B.100d.txt'

#Defining a function to create an array of word coefs
def get_coefs(word, *arr): 
    return word, np.asarray(arr, dtype='float32')

#Storing word embeddings in a dictionary
embeddings_dict = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(embedding_file))

#Stacking embeddings together
all_emb = np.stack(embeddings_index.values())
emb_mean = all_embs.mean() #Calculating means
emb_std =  all_embs.std() #Calculating stds
embed_size = all_embs.shape[1] #Calculating embedding size

#Indexing words
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))

#Creating an embedding matrix 
embedding_matrix = embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

#Setting model params
batch_size = 256 #controls the accuracy of the estimate of the error gradient, number of divisions of the dataset
epochs = 5 #one epoch is when an entire dataset is passed forward and backward through a nn only ONCE
embed_size = 100

#multiple epochs pass the dataset through a nn multiple times
#as he number of epoch's increases the more times the weights are changed in the nn 
#and the curve goes from underfitting to overfitting the data

#an iteration is the number of batches needed to complete one epoch
#ie for a dataset of 500 split into batches of 100, it will take 5 iterations to complete 1 epoch
#gradient descent is an iterative optimization algo used in ML to find the minima of a curve

#the learning rate is a tuning parameter in an optimization algo that determined the step size at each iteration
#while moving toward a minimum of a loss function

#trade off between convergence and overshooting, too high will make the learning jump over the minima and
#too low will either take too long to converge or get stuck in a local minima
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

#Defining NN params
max_features = 500 #number of words to consider as features
maxlen = 250 #cuts of text after this many words

#Defining Neural Network
model = Sequential()

#Non-trainable embeddidng layer
model.add(Embedding(max_features, output_dim=embed_size, weights=[embedding_matrix], input_length=maxlen, trainable=False))

#LSTM 
model.add(LSTM(units=128 , return_sequences = True , recurrent_dropout = 0.25 , dropout = 0.25))
model.add(LSTM(units=64 , recurrent_dropout = 0.1 , dropout = 0.1))
model.add(Dense(units = 32 , activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=keras.optimizers.Adam(lr = 0.01), loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train, batch_size = batch_size , validation_data = (X_val,y_val) , epochs = epochs , callbacks = [learning_rate_reduction])

print("Accuracy of the model on Training Data is:" , model.evaluate(X_train,y_train)[1]*100)
print("Accuracy of the model on Validation Data is:" , model.evaluate(X_val,y_val)[1]*100)

#Extracting results for accuracy and loss
epochs = [i for i in range(10)]
train_acc = history.history['acc']
train_loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

#Visualizing Training & Validation Accuracy & Loss

#Accuracy Plot
fig1 = plt.plot(figsize=(20,10))
plt.plot(epochs , train_acc , 'bo-' , label = 'Training Accuracy')
plt.plot(epochs , val_acc , 'yo-' , label = 'Testing Accuracy')
plt.title('Training & Testing Accuracy')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

#Loss Plot
fig2 = plt.plot(figsize=(20,10))
plt.plot(epochs , train_loss , 'ro-' , label = 'Training Loss')
plt.plot(epochs , val_loss , 'go-' , label = 'Testing Loss')
plt.title('Training & Testing Loss')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

#Classification report
print(classification_report(y_val, pred, target_names = ['Fake','Not Fake']))
