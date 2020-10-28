import pandas as pd
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model,load_model
import numpy as np
from pickle import dump,load
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import string
import collections
from keras import Input
from keras.layers import Dropout,Dense,Embedding,LSTM,add
import cv2
import matplotlib.pyplot as plt #pip install matplotlib



filename = "D:/project/Image_Captioning/flickr30k_images/flickr30k_images/results.csv"
file = open(filename, 'r',encoding="mbcs")
doc = file.read()

image_rec  = [] # Record of All images
All_desc = [] # Record of All descriptions
descriptions = dict() # Dictionatry of image -> list of All description
for line in doc.split('\n'):
    # split line by white space
    # print(line)
    if line == "age_name| comment_number| comment" or line == "image_name| comment_number| comment":
        continue
    tokens = line.split("|")
    image_rec.append(tokens[0])
    All_desc.append(tokens[-1])
    # print(tokens)
    if len(line) < 2:
        continue

    # take the first token as image id, the rest as description
    image_id, image_desc = tokens[0], tokens[-1]
    # extract filename from image id
    image_id = image_id.split('.')[0]
    # convert description tokens back to string
    if image_id not in descriptions :
        descriptions[image_id] = list()
    descriptions[image_id].append(image_desc)

    # print(image_id)
# print(descriptions)


def prepare_data():

    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()

            # convert to lower case
            desc = [word.lower() for word in desc]
            # remove punctuation from each token
            # desc.translate(str.maketrans('', '', string.punctuation))
            desc = [s.translate(str.maketrans('', '', string.punctuation)) for s in desc]
            # remove hanging 's' and 'a'
            desc = [word for word in desc if len(word) > 1]
            # remove tokens with numbers in them
            desc = [word for word in desc if word.isalpha()]
            # store as string
            desc.insert(0,"<start>")
            desc.append("<end>")
            # print(desc)
            desc_list[i] = ' '.join(desc)
train_X,train_y = [],[]
# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    batches = []
    count = 0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            if count == 10000:
                break
            count+=1
            # retrieve the photo feature
            photo = photos[key]
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            print(count)
            # yield the batch data
            if n==num_photos_per_batch:
                train_X.append([np.array(X1),np.array(X2)])
                train_y.append(np.array(y))
                print("batches",len(batches))
                n=0
                break
            
        break
        return batches


def embedding():
        # Load Glove vectors
    glove_dir = 'D:/project/Image_Captioning/glove_6B/glove.6B.200d.txt'
    embeddings_index = {} # empty dictionary
    f = open(glove_dir, encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
    #    print(coefs)
        embeddings_index[word] = coefs
    f.close()
    
    vocab_size = len(vocabulary)+1
    embedding_dim = 200
    # Get 200-dim dense vector for each of the 10000 words in out vocabulary
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    print(wordtoix)
    for word, i in wordtoix.items():
        #if i < max_words:
    #    print(i,word)
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in the embedding index will be all zeros
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

#class attention(Layer):
#    def __init__(self,**kwargs):
#        super(attention,self).__init__(**kwargs)
#
#    def build(self,input_shape):
#        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
#        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
#        super(attention, self).build(input_shape)
#
#    def call(self,x):
#        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
#        at=K.softmax(et)
#        at=K.expand_dims(at,axis=-1)
#        output=x*at
#        return K.sum(output,axis=1)
#
#    def compute_output_shape(self,input_shape):
#        return (input_shape[0],input_shape[-1])
#
#    def get_config(self):
#        return super(attention,self).get_config()
    
def model(vocab_size,embedding_dim,max_length):
    # image feature extractor model
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    # partial caption sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    # decoder (feed forward) model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # merge the two input models
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    return model

def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features  

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r',encoding="mbcs")
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		if line == "age_name| comment_number| comment" or line == "image_name| comment_number| comment":
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)


prepare_data()
embedding_dim = 200
#Get max length of description
max_length = max(len(d.split()) for d in All_desc)


#Create Vocabulary
vocabulary = set() # Word Vocabulary
counter = {} # Word counter
for key in descriptions.keys():
    lst = []
    for d in descriptions[key]:
        # vocabulary.update(d.split())
        # [vocabulary.update(d.split()) for d in descriptions[key]]
        for word in d.split():
            if word in counter:
                counter[word] += 1
            else:
                counter[word] = 1

for word in counter:
    if counter[word] >= 5:
        vocabulary.add(word)
        
#print(len(vocabulary))
#vocab_size = len(vocabulary)+1
#length_df = pd.DataFrame.from_dict(counter,orient='index',
#                       columns=['description'])
#length_df.plot(kind = 'scatter')

ixtoword = {}
wordtoix = {}
ix = 1
for w in vocabulary:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1




embedding_matrix = embedding()
print(len(embedding_matrix))
print(vocab_size)
# Building Model
#model = model(vocab_size,embedding_dim,max_length)
#model.layers[2].set_weights([embedding()])
#model.layers[2].trainable = False
#
#model.compile(loss='categorical_crossentropy', optimizer='adam')
#
#model.fit(train_X[0],train_y[0],epochs=20, verbose=1)
#model.save("D:/project/Image_Captioning/model_without_Attention.h5")
#
#filename = 'D:/project/Image_Captioning/flickr30k_images/flickr30k_images/results.csv'
#train = load_set(filename)
#photos = load_photo_features("D:/project/Image_Captioning/features.pkl",train)
#s = data_generator(descriptions, photos, wordtoix, max_length, 5)

#print(s)
#model.fit([])








def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        print(max_length)
        sequence = pad_sequences([sequence], maxlen=max_length)
        print(photo.shape)
        yhat = model.predict([photo,sequence], verbose=1)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[:-1]
    final = ' '.join(final)
    return final

#image = load_image()
#print(image.shape)
#greedySearch(image)






print(len(vocabulary))




















inception = InceptionV3(weights='imagenet')
inception = Model(inception.input, inception.layers[-2].output)

def load_image():
        filename = "C:/Users/DELL/Pictures/boy-girl meme.jpg"
        
        image = load_img(filename, target_size=(299, 299))
        #print(image)
    	# convert the image pixels to a numpy array
        image = img_to_array(image)
        #print(image.shape)
    	# reshape data for the model
        image = np.expand_dims(image, axis=0)
        #print(image.shape)
    	# prepare the image for the VGG model
        image = preprocess_input(image)
        #print(image.shape)
        image = inception.predict(image)
        print(image.shape)
        # get features
        image = np.reshape(image, image.shape[1])
        return image

#features = dict()
#
#for img in image_rec:
#    
#    try:
#        if len(img) == 0:
#            continue
#        filename = "/home/shivansh/Desktop/projects/Concordia_assignment/Image_Captioning/flickr30k_images/flickr30k_images" + '/' + img
#        
#        print(img)
#        image = load_img(filename, target_size=(299, 299))
#        #print(image)
#    	# convert the image pixels to a numpy array
#        image = img_to_array(image)
#        #print(image.shape)
#    	# reshape data for the model
#        image = np.expand_dims(image, axis=0)
#        #print(image.shape)
#    	# prepare the image for the VGG model
#        image = preprocess_input(image)
#        #print(image.shape)
#        image = model.predict(image)
#        print(image.shape)
#        # get features
#        image = np.reshape(image, image.shape[1])
#    
#        # get image id
#        image_id = img.split('.')[0]
#        # store feature
#        features[image_id] = image
#    except:
#        print("exception in ",img, "| length of img: ",len(img))
#        break
#    
#
##print(features)
#dump(features, open('features.pkl', 'wb'))
def inference(image_path,model):    
    
    x1,x2 = [],[]
    image = load_img(image_path, target_size=(299, 299))

   	# convert the image pixels to a numpy array
    image = img_to_array(image)
    print(image.shape)
#   	# reshape data for the model
    image = np.expand_dims(image, axis=0)
    print(image.shape)
#    # prepare the image for the VGG model
    image = preprocess_input(image)
    print(image.shape)
    image = inception.predict(image)
#    # get features
    image = np.reshape(image, image.shape[1])
    print(image.shape)
    x1.append(image)
#    image = np.array(image)
#    print(image.shape)
    in_text = '<start>'
	# iterate over the whole length of the sequence
    for i in range(max_length):
		# integer encode input sequence
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        print(sequence)        
		# pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
#        sequence = np.array(sequence)
        print(sequence.shape)
		# predict next word
        yhat = model.predict([x1,sequence], verbose=1)
		# convert probability to integer
        yhat = np.argmax(yhat)
		# map integer to word
        word = ixtoword[yhat]
		# stop if we cannot map the word
        if word is None:
            break		# append as input for generating the next word
        in_text += ' ' + word
		# stop if we predict the end of the sequence
        if word == '<end>':
            break
        print(in_text)    
#    in_text = 'startseq'
#    for i in range(max_length):
#        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
#        print(max_length)
#        sequence = pad_sequences([sequence], maxlen=max_length)
#        print(image.shape)
#        print(sequence)
##        print(np.argmax(sequence))
##            continue
#        try:
#            yhat = model.predict([image,sequence], verbose=1)
#        except:
#            continue
#        print(sequence)
#        yhat = np.argmax(yhat)
#        word = ixtoword[yhat]
#        in_text += ' ' + word
#        if word == 'endseq':
#            break
#    final = in_text.split()
#    final = final[:-1]
#    final = ' '.join(final)
#    print("final",final)
    
    
inception = InceptionV3(weights='imagenet')
inception = Model(inception.input, inception.layers[-2].output)
#inception.compile()
inception.layers[2].trainable = False

model_predict = model(vocab_size,embedding_dim,max_length)
model_predict.compile(loss='categorical_crossentropy', optimizer='adam')
model_predict.load_weights("D:/project/Image_Captioning/model_without_Attention.h5")
#model = load_model("D:/project/Image_Captioning/model_without_Attention.h5")
image_path = "C:/Users/DELL/Pictures/boy-girl meme.jpg"
inference(image_path,model_predict)



    




