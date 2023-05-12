import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
# nltk.download('punkt')


def tokenize(sentence):
  return nltk.word_tokenize(sentence)

stemmer = PorterStemmer()
def stem(word):
  return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
  # stem each word
  tokenized_sentence = [stem(word) for word in tokenized_sentence]
  # initialize bag with 0 for each word
  bag = np.zeros(len(all_words), dtype=np.float32)
  for idx, w in enumerate(all_words):
      if w in tokenized_sentence: 
          bag[idx] = 1.0

  return bag