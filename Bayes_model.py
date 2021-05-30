import numpy as np
import pandas as pd
from math import prod

# read data with pandas
data = pd.read_csv('datasets/spam_data/SMSSpamCollection', sep='\t', header=None, names=['Label', 'message'])

# lower case it and remove punctuation(note that this will reduce the model's accuracy)
data['message'] = data.message.str.lower()
data['message'] = data.message.str.replace('\W', ' ')

# divide train|test --> 80%|20%
train_data = data[:int(len(data)*0.8)]
test_data = data[int(len(data)*0.8):]

# the entire vocabulary will be ever word of the train set
voc = []
for message in train_data['message']:
    voc += message.split()
voc = set(voc)

spam_messages = train_data[train_data['Label'] == 'spam']
ham_messages = train_data[train_data['Label'] == 'ham']

# Laplace smoothing
a = 1

# n is number
n_word_per_spam = spam_messages['message'].apply(len)
n_word_per_ham = ham_messages['message'].apply(len)
n_spam = n_word_per_spam.sum()
n_ham = n_word_per_ham.sum()
n_voc = len(voc)

# prior probability that it is a spam or ham
P_spam = len(spam_messages) / len(train_data)
P_ham = len(ham_messages) / len(train_data)

# probability word at index i is in spam_messages
wi_spam = []

for wi in voc:
    count = 0
    for message in spam_messages.message:
        for word in message.split():
            if word == wi:
                count += 1
    wi_spam.append(count)
 
# probability w at index i is in ham_messages
wi_ham = []

for wi in voc:
    count = 0
    for message in ham_messages.message:
        for word in message.split():
            if word == wi:
                count += 1
    wi_ham.append(count)

# P( Spam | message )
def spam_probability(message):
    w_prob = 1
    
    for word in message.split():
        if word.lower() in voc:
            # since it's with respect to the vocabulary
            word_position = list(voc).index(word)
            n_wi = wi_spam[word_position]
            w_prob = w_prob * ((n_wi + a) / (n_spam + len(voc)))
        else:
            # not in voc
            w_prob = w_prob * (a / (n_spam + len(voc)))
    probability = P_spam * w_prob
    return probability

def ham_probability(message):
    w_prob = 1
    
    for word in message.split():
        if word.lower() in voc:
            word_position = list(voc).index(word)
            n_wi = wi_ham[word_position]
            w_prob = w_prob * (n_wi + a) / (n_ham + len(voc))
        else:
            w_prob = w_prob * (a / (n_ham + len(voc)))
    probability = P_ham * w_prob
    return probability

# classify a given message
def classification(message):    
    if spam_probability(message) > ham_probability(message):
        return 'spam'
    else:
        return 'ham'

def accuracy(test_set):
    precision = 0
    for i in range(len(test_set)):
        if classification(message) == test_set.Label.iloc[i]:
            precision += 1
    precision = precision / len(test_set)
    return precision
 

print(accuracy(test_data))
