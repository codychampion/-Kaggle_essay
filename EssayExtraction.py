import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import re
import nltk
from collections import Counter

TRAIN_CSV = "./train.csv"
TRAIN_DIR = "../input/feedback-prize-2021/train/"

# Read in documents.
# Separate essays into sentences.
# Words unique to the category
# Sentence compositions
# Sentiment
# Emotion
# Direction from overall document (more or less consistent e.g. counterpoint?)

train = pd.read_csv(TRAIN_CSV)

IDS = train.id.unique()

random.seed(2022)
essayID = random.choice(train["id"].unique())

one_essay = train[train["id"] == essayID]
print(one_essay.head())

###Calculating things for just one essay
file = open("./train/" + essayID + ".txt", "r")
essaytxt = file.read()
# splitEssay = re.split(" ""|\n", essaytxt)

##nltk things
tokens = nltk.sent_tokenize(essaytxt)
tokenw = nltk.word_tokenize(essaytxt, preserve_line=True)
wordPos = nltk.pos_tag(tokenw)

# chunking words
chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>}"""
chunkParser = nltk.RegexpParser(chunkGram)
chunked = chunkParser.parse(wordPos)


# print(chunked)

# Lead - an introduction that begins with a statistic, a quotation, a description, or some other device to grab the reader’s attention and point toward the thesis
# Position - an opinion or conclusion on the main question
# Claim - a claim that supports the position
# Counterclaim - a claim that refutes another claim or gives an opposing reason to the position
# Rebuttal - a claim that refutes a counterclaim
# Evidence - ideas or examples that support claims, counterclaims, or rebuttals.
# Concluding Statement - a concluding statement that restates the claims

# all_words = nltk.FreqDist(w.lower() for w in essaytxt.words()) # not currently working


def sent_features(essaytxt):
    features = {}
    features["sentPosition"] = tokens.index
    # features["wordPosition"] = tokenw.index
    # features["wordInSentPosition"] =
    for d in '0123456789':
        features["count(%s)" % d] = tokens.count(d)
        features["has(%s)" % d] = d in tokens
    features["sentQuoteCount"] = tokens.count('\"')
    features["sentNumberCount"] = sum(c.isdigit() for c in tokens)

    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count(%s)" % letter] = tokens.count(letter)
        features["has(%s)" % letter] = (letter in tokens)
    return features

def posCounter(sent, pos):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    count= Counter(tag for word, tag in sent)
    countList = count.items()
    try: # so an error doesn't disrupt the function.
        # index = [y[0] for y in countList].index(pos)
        return countList[index][1]
    except: # a zero is returned if an index(pos) is not found
        return 0

#Create a loop to extract similar information from several sentences in a lisusing the function defined above



NNPfreq=[]
for i in range(len(tokens)):
    score=posCounter(tokens[i], 'NNP')
    print(score)
    NNPfreq.append(score)

print(NNPfreq[:])
