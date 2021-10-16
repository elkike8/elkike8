import pandas as pd
import numpy as np
import re

# https://www.kaggle.com/currie32/a-south-park-chatbot
# https://www.youtube.com/watch?v=DItR-l59i6M&list=PLTuKYqpidPXbulRHl8HL7JLRQXwDlqpLO&index=5
# https://medium.com/swlh/how-to-design-seq2seq-chatbot-using-keras-framework-ae86d950e91d

southpark = pd.read_csv("C:/Users/FerGo/OneDrive/ACIT/2021/Computational Intelligence/Project/S2S South Park/All-seasons.csv")[:1000]

# define function to clean the text 

def clean_text(text):

    text = text.lower()
    
    text = re.sub(r"\n", "",  text)
    text = re.sub(r"[-()]", "", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r"\!", " ! ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\,", " , ", text)
    text = re.sub(r"\"", " \" ", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)

    
    return text

# Clean the scripts and add them to the same list.
text = []

for line in southpark.Line:
    text.append(clean_text(line))
    
# Find the length of lines in number of words + Characters
lengths = []
for line in text:
    lengths.append(len(line.split()))

# Create a dataframe so that the values can be inspected
lengths = pd.DataFrame(lengths, columns=['counts'])


# Using lines shorter than the 95 percentile
max_line_length = int(np.percentile(lengths, 95) - np.percentile(lengths, 95)%10)

short_text = []
for line in text:
    if len(line.split()) <= max_line_length:
        short_text.append(line)

# Create the questions and answers texts.
# The answer text is the line following the question text.
q_text = short_text[:-1]
a_text = short_text[1:]

# adding tokens to sentences
for i in range(len(a_text)):
    a_text[i] = "<BOS> " + a_text[i] + " <EOS>"

# make vocabulary
from tensorflow.keras.preprocessing.text import Tokenizer

VOCAB_SIZE = 10000
tokenizer = Tokenizer(num_words=VOCAB_SIZE, lower=False, filters="")
tokenizer.fit_on_texts(q_text + a_text)
dictionary = tokenizer.word_index

word2token = {}
token2word = {}

for k, v in dictionary.items():
    if v < VOCAB_SIZE:
        word2token[k] = v
        token2word[v] = k
    else:
        continue

VOCAB_SIZE = len(word2token)+1

# tokenizing words

encoder_seq = tokenizer.texts_to_sequences(q_text)
decoder_seq = tokenizer.texts_to_sequences(a_text)

# padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

encoder_input = pad_sequences(encoder_seq, maxlen = max_line_length, padding="post", truncating="post")
decoder_input = pad_sequences(decoder_seq, maxlen = max_line_length, padding="post", truncating="post")

# one-hot encoded answers
# remove first word (<BOS>)
for i in range(len(decoder_seq)):
    decoder_seq[i]=decoder_seq[i][1:]
#pad with 0
padded_answers = pad_sequences(decoder_seq,
                               maxlen = max_line_length,
                               padding="post")
decoder_output = to_categorical(padded_answers, VOCAB_SIZE)

#deleting non-necessary variables
del(a_text, decoder_seq, encoder_seq, dictionary, i, k, line, padded_answers, q_text, short_text, v)

# =============================================================================
# Model
# =============================================================================
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input

# encoder will be used to capture space-dependent relations between words from the questions
# about 200 neurons needed 

enc_inputs = Input(shape=(None,))
enc_embeding = Embedding(VOCAB_SIZE, 200, mask_zero=True)
enc_embeding = enc_embeding(enc_inputs)
enc_lstm = LSTM(200,  return_state=True)
enc_outputs, h, c = enc_lstm(enc_embeding)
enc_states = [h, c]

# decoder will be used to capture space-dependent relations between words from the answers using encoder's internal state as a context

dec_inputs = Input(shape=(None,))
dec_embedding = Embedding(VOCAB_SIZE, 200, mask_zero=True)
dec_embedding = dec_embedding(dec_inputs)
dec_lstm = LSTM(200, return_state=True, return_sequences=True)
dec_outputs, _, _ = dec_lstm(dec_embedding, initial_state = enc_states)

# decoder is connected to the output Dense layer
dec_dense = Dense(VOCAB_SIZE, activation = "softmax")
output = dec_dense(dec_outputs)

model = Model([enc_inputs, dec_inputs], output)
# output of this network will look like this:
# y_true = [0.05, 0.95, 0...]
# and expected one-hot encoded output like this:
# y_pred = [0, 1, 0...]
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["acc"])
model.summary()

model.fit([encoder_input, decoder_input],
          decoder_output,
          batch_size=50,
          epochs = 200)

model.save("OneDrive/ACIT/2021/Computational Intelligence/Project/S2S South Park/Model/sb1.h5")

# =============================================================================
# Bot
# =============================================================================

model.load_weights("OneDrive/ACIT/2021/Computational Intelligence/Project/S2S South Park/Model/sb1.h5")

def make_inference_models():
    # two inputs for the state vectors returned by encoder
    dec_state_input_h = Input(shape=(200,))
    dec_state_input_c = Input(shape=(200,))
    dec_states_inputs = [dec_state_input_h, dec_state_input_c]
    # these state vectors are used as an initial state 
    # for LSTM layer in the inference decoder
    # third input is the Embedding layer as explained above   
    dec_outputs, h, c = dec_lstm(dec_embedding,
                                    initial_state=dec_states_inputs)
    dec_states = [h, c]
    # Dense layer is used to return OHE predicted word
    dec_outputs = dec_dense(dec_outputs)
    dec_model = Model(
        inputs=[dec_inputs] + dec_states_inputs,
        outputs=[dec_outputs] + dec_states)
   
    # single encoder input is a question, represented as a sequence 
    # of integers padded with zeros
    enc_model = Model(inputs=enc_inputs, outputs=enc_states)
   
    return enc_model, dec_model

enc_model, dec_model = make_inference_models()

def str_to_tokens(sentence: str):

    sentence = clean_text(sentence)
    words = sentence.lower().split()
    tokens_list = list()
    for current_word in words:
        result = tokenizer.word_index.get(current_word, '')
        if result != "":
            tokens_list.append(result)
    return pad_sequences([tokens_list],
                         maxlen = max_line_length,
                         padding = "post")

# chatting loop

for _ in range(5):
    # encode the input sequence into state vectors
    states_values = enc_model.predict(str_to_tokens(input("user : ")))
    # start with a target sequence of size 1 - word 'start'   
    empty_target_seq = np.zeros((1, 1))
    empty_target_seq[0, 0] = tokenizer.word_index["<BOS>"]
    stop_condition = False
    decoded_translation = ""
    while not stop_condition:
        # feed the state vectors and 1-word target sequence 
        # to the decoder to produce predictions for the next word
        dec_outputs, h, c = dec_model.predict([empty_target_seq] 
                                              + states_values)         
        # sample the next word using these predictions
        sampled_word_index = np.argmax(dec_outputs[0, -1, :])
        sampled_word = None
        # append the sampled word to the target sequence
        for word, index in tokenizer.word_index.items():
            if sampled_word_index == index:
                if word != "<EOS>":
                    decoded_translation += " {}".format(word)
                sampled_word = word
        # repeat until we generate the end-of-sequence word 'end' 
        # or we hit the length of answer limit
        if sampled_word == "<EOS>" \
                or len(decoded_translation.split()) \
                > max_line_length:
            stop_condition = True
        # prepare next iteration
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = sampled_word_index
        states_values = [h, c]
    print("southbot: " + decoded_translation)
