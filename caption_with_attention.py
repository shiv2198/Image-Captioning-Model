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
from keras import Input
from keras.layers import Dropout,Dense,Embedding,LSTM,add,Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from math import ceil
#from Attention import Attention
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
 
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super().__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        print("Input shape[1]",input_shape)
        self.W = self.add_weight(shape = (input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape = (input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):

        features_dim = self.features_dim
        step_dim = self.step_dim
        print(features_dim,step_dim)
        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
        print(eij)
        if self.bias:
            eij += self.b

        eij = K.tanh(eij)
        a = K.exp(eij)
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
    
descriptions = dict() # Dictionatry of image -> list of All description

def create_description():
    filename = "D:/project/Image_Captioning/flickr30k_images/flickr30k_images/results.csv"
    file = open(filename, 'r',encoding="mbcs")
    doc = file.read()
    image_rec  = [] # Record of All images
    All_desc = []
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
    max_length = max(len(d.split()) for d in All_desc)
    return max_length

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

def caption_model(vocab_size,max_length):
    # image feature extractor model
    inputs1 = Input(shape=(2048,))
#    print(inputs1)
    fe1 = Dropout(0.5)(inputs1)
#    print(fe1)
    fe2 = Dense(128, activation='relu')(fe1)
#    print(fe2)
    # partial caption sequence model
    inputs2 = Input(shape=(max_length,))
    print(inputs2)
    se1 = Embedding(vocab_size, 128, mask_zero=True)(inputs2)
    print(se1)
    se2 = Dropout(0.5)(se1)
    print(se2)
    se3 = LSTM(128,return_sequences=True)(se2)
    print(se3)
    attention = Attention(max_length)(se3)
#    print(inputs1)
    # decoder (feed forward) model
    decoder1 = add([fe2, attention])
    decoder2 = Dense(128, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # merge the two input models
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
#    plot_model(model, to_file='model.png', show_shapes=True)
    return model
#model = model(vocab_size,max_length)
# data generator, intended to be used in a call to model.fit_generator()
def data_generator(tokenizer,data, max_length, batch):
    X1, X2, y = list(), list(), list()
    n=0
#    count = 0
    # loop for ever over images
    print(len(data))
    while 1:
        for index, row in data.iterrows():
    #        print(index,row['description'],row['image'])
    #         retrieve the photo feature
            photo = row['image']
            for desc in row['description']:
                # encode the sequence
                desc = [word for word in desc.split(' ') if word in vocabulary]
                seq = tokenizer.texts_to_sequences([desc])[0]
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
            n+=1
            # yield the batch data
            if n==batch:
#                print("\t",index)
#                train_X = [np.array(X1),np.array(X2)]
#                train_y = np.array(y)
                yield [[np.array(X1), np.array(X2)], np.array(y)]
                X1, X2, y = list(), list(), list()
                n=0
#        return [train_X,train_y]


model = None
inception = InceptionV3(weights='imagenet')
inception = Model(inception.input, inception.layers[-2].output)
inception.layers[2].trainable = False
def create_image_features(image_rec):
    features = dict()
    for img in image_rec:
        try:
            if len(img) == 0:
                continue
            filename = "/home/shivansh/Desktop/projects/Concordia_assignment/Image_Captioning/flickr30k_images/flickr30k_images" + '/' + img
            
            print(img)
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
            image = model.predict(image)
            print(image.shape)
            # get features
            image = np.reshape(image, image.shape[1])
        
            # get image id
            image_id = img.split('.')[0]
            # store feature
            features[image_id] = image
        except:
            print("exception in ",img, "| length of img: ",len(img))
            break
        
    
    #print(features)
    dump(features, open('features.pkl', 'wb'))


# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features        

# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc    

def create_tokenizer(descriptions):
#	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(vocabulary)
	return tokenizer

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

#Create Vocabulary
vocabulary = set() # Word Vocabulary
def create_vocab():
    counter = {} # Word counter
    for key in descriptions.keys():
        for d in descriptions[key]:
    #         vocabulary.update(d.split())
    #        [vocabulary.update(d.split()) for d in descriptions[key]]
            for word in d.split():
                if word in counter:
                    counter[word] += 1
                else:
                    counter[word] = 1
#    print(len(counter))
    for word in counter:
        if counter[word] >= 10:
            vocabulary.add(word)
    return vocabulary
def load_image(filename):
        
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

def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None    


def inference(image_path,model_path,tokenizer):    
    # Building Model
    inception = InceptionV3(weights='imagenet')
    inception = Model(inception.input, inception.layers[-2].output)
    #inception.compile()
    inception.layers[2].trainable = False
    print(model)
    model_predict = None
    if not model:
        model_predict = caption_model(vocab_size,max_length)
        model_predict.compile(loss='categorical_crossentropy', optimizer='adam')
        model_predict.load_weights(model_path)
        #model = load_model("D:/project/Image_Captioning/model_without_Attention.h5")
    
    x1 = []
    image = load_img(image_path,target_size=(299, 299))

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
    # get features
    image = np.reshape(image, image.shape[1])
    x1.append(image)
    print(image.shape)
    in_text = '<start>'
	# iterate over the whole length of the sequence
    for i in range(max_length):
		# integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
        yhat = model_predict.predict([x1,sequence], verbose=0)
		# convert probability to integer
        yhat = np.argmax(yhat)
		# map integer to word
        word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
        if word is None:
            continue		# append as input for generating the next word
        in_text += ' ' + word
		# stop if we predict the end of the sequence
        if word == '<end>':
            break
    print(in_text)
        
def train():
    data_length = int(len(dataframe)*0.7)
    train = dataframe[:data_length]
    test = dataframe[data_length:]
    
#    filepath = "caption_model_trial.h5"
#    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    epochs = 20
    batch_size = 64
    steps_per_epoch = ceil(len(train) / batch_size)
    val_steps = ceil(len(test) / 64)
    #for i in range(epochs):
    #    train_x,train_y = data_generator(tokenizer,train, max_length, batch_size)
    model = caption_model(vocab_size,max_length)
    model.fit_generator(data_generator(tokenizer,train, max_length, batch_size), 
                        epochs=epochs,steps_per_epoch = steps_per_epoch, verbose=1,
                        validation_data=data_generator(tokenizer,test, max_length, batch_size),
                        validation_steps = val_steps)

    model.save("Attention_Caption_model.h5")
    return model
#Get max length of description and create description
max_length = create_description()
vocabulary = create_vocab()
prepare_data()
#print(vocabulary)

tokenizer = create_tokenizer(descriptions)
vocab_size = len(vocabulary) + 1


filename = 'D:/project/Image_Captioning/flickr30k_images/flickr30k_images/results.csv'
data = load_set(filename)

photos = load_photo_features("D:/project/Image_Captioning/features.pkl",data)


ds = [descriptions, photos]
data = {}
for k in descriptions.keys():
    if k not in ds[0] or k not in ds[1]:
        continue
    data[k] = tuple([d[k] for d in ds])
    
dataframe = pd.DataFrame.from_dict(data,orient='index',
                       columns=['description','image'])


model = train()


image_path = image_path = "C:/Users/DELL/Pictures/boy-girl meme.jpg"
model_path = ("D:/project/Image_Captioning/Attention_Caption_model.h5")
inference(image_path,model_path,tokenizer)














#data_length = int(len(dataframe)*0.7)
#train = dataframe[:data_length]
#test = dataframe[data_length:]
#
#filepath = "caption_model_trial.h5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#epochs = 1
#batch_size = 64
#steps_per_epoch = ceil(len(train) / batch_size)
#val_steps = ceil(len(test) / 64)
##for i in range(epochs):
##    train_x,train_y = data_generator(tokenizer,train, max_length, batch_size)
#model = caption_model(vocab_size,max_length)
#model.fit_generator(data_generator(tokenizer,train, max_length, batch_size), 
#                    epochs=1,steps_per_epoch = steps_per_epoch, verbose=1, callbacks=[checkpoint],
#                    validation_data=data_generator(tokenizer,test, max_length, 500),
#                    validation_steps = val_steps)
#
#model.save("mode_trial.h5")




