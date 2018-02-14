from numpy import asarray
import pandas as pd
import numpy as np
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.utils import np_utils
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


dataset = pd.read_csv('train.tsv', delimiter = '\t', quoting = 3)

dataset.drop('SentenceId', axis=1, inplace = True)

lengths = []
for i in range(0,156060):
    msg = ''
    if msg == '':
        msg = len(dataset['Phrase'][i].split())
        lengths.append(msg)

lengths = np.array(lengths)
dataset['words'] = lengths
dataset['words'].describe()

#analysis part 

dataset = dataset.sort_values(by = 'words')
dataset['words'].value_counts()
dataset = dataset.reset_index()
dataset.drop('index', inplace = True, axis=1)
dataset['words'].value_counts()

#buiding different dataset for the different length of reviews.(to avoid sparsity form padding)
bucket_1 =dataset[dataset['words'] == 1].reset_index() 
bucket_1.drop('index', inplace=True,axis=1) 

bucket_2 =dataset[dataset['words'] == 2].reset_index()
bucket_2.drop('index', inplace=True,axis=1) 

bucket_3 =dataset[dataset['words'] == 3].reset_index()
bucket_3.drop('index', inplace=True,axis=1) 

bucket_4 =dataset[dataset['words'] == 4].reset_index() 
bucket_4.drop('index', inplace=True,axis=1)

bucket_5 =dataset[dataset['words'] == 5].reset_index() 
bucket_5.drop('index', inplace=True,axis=1)

bucket_6 =dataset[dataset['words'] == 6].reset_index() 
bucket_6.drop('index', inplace=True,axis=1)

bucket_7_12 =dataset[(dataset['words'] > 6) & (dataset['words'] <= 12)].reset_index() 
bucket_7_12.drop('index', inplace=True,axis=1)

bucket_13_18 =dataset[(dataset['words'] > 12) & (dataset['words'] <= 18)].reset_index() 
bucket_13_18.drop('index', inplace=True,axis=1)

bucket_19_24 =dataset[(dataset['words'] > 18) & (dataset['words'] <= 24)].reset_index() 
bucket_19_24.drop('index', inplace=True,axis=1)

bucket_25_30 =dataset[(dataset['words'] > 24) & (dataset['words'] <= 30)].reset_index() 
bucket_25_30.drop('index', inplace=True,axis=1)

bucket_31_36 =dataset[(dataset['words'] > 30) & (dataset['words'] <= 36)].reset_index() 
bucket_31_36.drop('index', inplace=True,axis=1)

bucket_37_42 =dataset[(dataset['words'] > 36) & (dataset['words'] <= 42)].reset_index() 
bucket_37_42.drop('index', inplace=True,axis=1)

bucket_43_53 =dataset[(dataset['words'] > 42) & (dataset['words'] <= 53)].reset_index() 
bucket_43_53.drop('index', inplace=True,axis=1)

def create_labels(df):
    labels = []
    labels = df['Sentiment'].values
    labels = np_utils.to_categorical(labels, 5)
    df.drop('Sentiment', axis=1, inplace = True)
    return labels

    
bucket_1_labels = create_labels(bucket_1)
bucket_2_labels = create_labels(bucket_2)
bucket_3_labels = create_labels(bucket_3)   
bucket_4_labels = create_labels(bucket_4)   
bucket_5_labels = create_labels(bucket_5)   
bucket_6_labels = create_labels(bucket_6)   
bucket_7_12_labels = create_labels(bucket_7_12)    
bucket_13_18_labels = create_labels(bucket_13_18)    
bucket_19_24_labels = create_labels(bucket_19_24)   
bucket_25_30_labels = create_labels(bucket_25_30)   
bucket_31_36_labels = create_labels(bucket_31_36)   
bucket_37_42_labels = create_labels(bucket_37_42)   
bucket_43_53_labels = create_labels(bucket_43_53)   


embeddings_index = dict()
f = open('glove.6B.100d.txt',encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# text cleaning
def make_all_corpus(df, length):
    corpus = []
    for i in range(0, len(df)):
        review = re.sub('[^a-zA-Z]', ' ', df['Phrase'][i])
        review = review.lower()
        review = review.split()
        #ps = PorterStemmer()
        wordnet_lemmatizer = WordNetLemmatizer()
        review = [wordnet_lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    t = Tokenizer()
    t.fit_on_texts(corpus)
    vocab_size = len(t.word_index) + 1
    encoded_corpus = t.texts_to_sequences(corpus)
    #print(encoded_corpus)
    
    max_length = length
    padded_corpus = pad_sequences(encoded_corpus, maxlen=max_length, padding='post')
    embedding_matrix = zeros((vocab_size, 100))
    for word, i in t.word_index.items():
    	embedding_vector = embeddings_index.get(word)
    	if embedding_vector is not None:
    		embedding_matrix[i] = embedding_vector

    return padded_corpus, embedding_matrix, vocab_size

        

bucket_1_corpus, bucket_1_embedding_matrix, bucket_1_vocab_size = make_all_corpus(bucket_1, 1) 

bucket_2_corpus, bucket_2_embedding_matrix, bucket_2_vocab_size = make_all_corpus(bucket_2, 2) 

bucket_3_corpus, bucket_3_embedding_matrix, bucket_3_vocab_size = make_all_corpus(bucket_3, 3) 

bucket_4_corpus, bucket_4_embedding_matrix, bucket_4_vocab_size = make_all_corpus(bucket_4, 4) 

bucket_5_corpus, bucket_5_embedding_matrix, bucket_5_vocab_size = make_all_corpus(bucket_5, 5) 

bucket_6_corpus, bucket_6_embedding_matrix, bucket_6_vocab_size = make_all_corpus(bucket_6, 6) 

bucket_7_12_corpus, bucket_7_12_embedding_matrix, bucket_7_12_vocab_size = make_all_corpus(bucket_7_12, 12) 

bucket_13_18_corpus, bucket_13_18_embedding_matrix, bucket_13_18_vocab_size = make_all_corpus(bucket_13_18, 18) 

bucket_19_24_corpus, bucket_19_24_embedding_matrix, bucket_19_24_vocab_size = make_all_corpus(bucket_19_24, 24) 

bucket_25_30_corpus, bucket_25_30_embedding_matrix, bucket_25_30_vocab_size = make_all_corpus(bucket_25_30, 30) 

bucket_31_36_corpus, bucket_31_36_embedding_matrix, bucket_31_36_vocab_size = make_all_corpus(bucket_31_36, 36) 

bucket_37_42_corpus, bucket_37_42_embedding_matrix, bucket_37_42_vocab_size = make_all_corpus(bucket_37_42, 42) 

bucket_43_53_corpus, bucket_43_53_embedding_matrix, bucket_43_53_vocab_size = make_all_corpus(bucket_43_53, 53) 

# model_creation

def create_model_all(padded_corpus,embedding_matrix,vocab_size,max_length,labels):
    
    model = Sequential()
    e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Flatten())
    model.add(Dense(32,init = 'uniform', activation='relu'))
    #model.add(Dense(,init = 'uniform', activation='relu'))
    model.add(Dense(64, init = 'uniform', activation = 'relu'))
    
    model.add(Dense(5,init = 'uniform', activation='sigmoid'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
       
    model.fit(padded_corpus, labels,epochs=50, verbose=2)
    
    return model

model_bucket_1 = create_model_all(bucket_1_corpus, bucket_1_embedding_matrix,
                                  bucket_1_vocab_size, 1, bucket_1_labels
                                  )
model_bucket_1.save_weights("model_bucket_1.h5")

model_bucket_2 = create_model_all(bucket_2_corpus, bucket_2_embedding_matrix,
                                  bucket_2_vocab_size, 2, bucket_2_labels
                                  )
model_bucket_2.save_weights("model_bucket_2.h5")

model_bucket_3 = create_model_all(bucket_3_corpus, bucket_3_embedding_matrix,
                                  bucket_3_vocab_size, 3, bucket_3_labels
                                  )
model_bucket_3.save_weights("model_bucket_3.h5")

model_bucket_4 = create_model_all(bucket_4_corpus, bucket_4_embedding_matrix,
                                  bucket_4_vocab_size, 4, bucket_4_labels
                                  )
model_bucket_4.save_weights("model_bucket_4.h5")

model_bucket_5 = create_model_all(bucket_5_corpus, bucket_5_embedding_matrix,
                                  bucket_5_vocab_size, 5, bucket_5_labels
                                  )
model_bucket_5.save_weights("model_bucket_5.h5")

model_bucket_6 = create_model_all(bucket_6_corpus, bucket_6_embedding_matrix,
                                  bucket_6_vocab_size, 6, bucket_6_labels
                                  )
model_bucket_6.save_weights("model_bucket_6.h5")

model_bucket_7_12 = create_model_all(bucket_7_12_corpus, bucket_7_12_embedding_matrix,
                                  bucket_7_12_vocab_size, 12, bucket_7_12_labels
                                  )
model_bucket_7_12.save_weights("model_bucket_7_12.h5")

model_bucket_13_18 = create_model_all(bucket_13_18_corpus, bucket_13_18_embedding_matrix,
                                  bucket_13_18_vocab_size, 18, bucket_13_18_labels
                                  )
model_bucket_13_18.save_weights("model_bucket_13_18.h5")

model_bucket_19_24 = create_model_all(bucket_19_24_corpus, bucket_19_24_embedding_matrix,
                                  bucket_19_24_vocab_size, 24, bucket_19_24_labels
                                  )
model_bucket_19_24.save_weights("model_bucket_19_24.h5")

model_bucket_25_30 = create_model_all(bucket_25_30_corpus, bucket_25_30_embedding_matrix,
                                  bucket_25_30_vocab_size, 30, bucket_25_30_labels
                                  )
model_bucket_25_30.save_weights("model_bucket_25_30.h5")

model_bucket_31_36 = create_model_all(bucket_31_36_corpus, bucket_31_36_embedding_matrix,
                                  bucket_31_36_vocab_size, 36, bucket_31_36_labels
                                  )
model_bucket_31_36.save_weights("model_bucket_31_36.h5")

model_bucket_37_42 = create_model_all(bucket_37_42_corpus, bucket_37_42_embedding_matrix,
                                  bucket_37_42_vocab_size, 42, bucket_37_42_labels
                                  )
model_bucket_37_42.save_weights("model_bucket_37_42.h5")

model_bucket_43_53 = create_model_all(bucket_43_53_corpus, bucket_43_53_embedding_matrix,
                                  bucket_43_53_vocab_size, 53, bucket_43_53_labels
                                  )
model_bucket_43_53.save_weights("model_bucket_43_53.h5")

#### for test #####

dataset_test = pd.read_csv('test.tsv', delimiter = '\t', quoting = 3)

dataset_test.drop('SentenceId', axis=1, inplace = True)

lengths_test = []
for i in range(0,66292):
    msg = ''
    if msg == '':
        msg = len(dataset_test['Phrase'][i].split())
        lengths_test.append(msg)

lengths_test = np.array(lengths_test)
dataset_test['words'] = lengths_test      
len(dataset_test)

len_dic=dict(zip(list(range(1,54)),[1,2,3,4,5,6,
                                    12,12,12,12,12,12,
                                    18,18,18,18,18,18,
                                    24,24,24,24,24,24,
                                    30,30,30,30,30,30,
                                    36,36,36,36,36,36,
                                    42,42,42,42,42,42,
                                    53,53,53,53,53,53,53,53,53,53,53
                                    ]))
 
   
y_pred = []

for j in range(0,len(dataset_test)):       
    
    review = dataset_test['Phrase'][j]
    corpus = [] 
    rev = re.sub('[^a-zA-Z]', ' ', review)
    rev = rev.lower()
    rev = rev.split()
    #ps = PorterStemmer()
    wordnet_lemmatizer = WordNetLemmatizer()
    rev = [wordnet_lemmatizer.lemmatize(word) for word in rev if not word in set(stopwords.words('english'))]
    rev = ' '.join(rev)
    corpus.append(rev)
    t = Tokenizer()
    t.fit_on_texts(corpus)
    vocab_size = len(t.word_index) + 1
    encoded_corpus = t.texts_to_sequences(corpus)
    #print(encoded_corpus)
    
    max_length = len_dic[vocab_size]  
    padded_corpus = pad_sequences(encoded_corpus, maxlen=max_length,
                                  padding='post')
    if max_length == 1:
        prediction = model_bucket_1.predict(padded_corpus)
    
    elif max_length == 2:
        prediction = model_bucket_2.predict(padded_corpus)
    
    elif max_length == 3:
        prediction = model_bucket_3.predict(padded_corpus)
    
    elif max_length == 4:
        prediction = model_bucket_4.predict(padded_corpus)
    
    elif max_length == 5:
        prediction = model_bucket_5.predict(padded_corpus)
       
    elif max_length == 6:
        prediction = model_bucket_6.predict(padded_corpus)
        
    elif max_length == 12:
        prediction = model_bucket_7_12.predict(padded_corpus)
    
    elif max_length == 18:
        prediction = model_bucket_13_18.predict(padded_corpus)
    
    elif max_length == 24:
        prediction = model_bucket_19_24.predict(padded_corpus)
    
    elif max_length == 30:
        prediction = model_bucket_25_30.predict(padded_corpus)
    
    elif max_length == 36:
        prediction = model_bucket_31_36.predict(padded_corpus)
    
    elif max_length == 42:
        prediction = model_bucket_37_42.predict(padded_corpus)
    
    else:
        prediction = model_bucket_43_53.predict(padded_corpus)
    
    p = prediction.argmax(axis = 1)
    
    y_pred.append(p)
    
   

y_sol = y_pred

y_sol = np.array(y_sol)
 

df_output = pd.DataFrame()

df_output['PhraseId'] = dataset_test['PhraseId']
df_output['Sentiment'] = y_sol
df_output[['PhraseId','Sentiment']].to_csv('output_batch.csv',index=False)
