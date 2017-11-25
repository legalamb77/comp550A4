from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stops = stopwords.words('english')
import sys
import glob

wnl = WordNetLemmatizer()
LIMIT = 25

'''
Takes a list of sentences, which are each a list of word tokens, and lemmatizes each of the words.
'''
def lemmatize(sentences):
    lemmatized = []
    for s in sentences:
        lemmatized.append([wnl.lemmatize(w) for w in s])
    return lemmatized

'''
Takes in a string (data), and applies standard preprocessing to it, returning a list of tokens.
'''
def preprocess(data):
    # Apply sentence segmentation
    sentences = sent_tokenize(data)
    # Apply tokenization within the sentences and store seperately, removing stop words and lowercasing all
    for i in range(len(sentences)):
        sentences[i] = sentences[i].lower()
        word_tokenized = word_tokenize(sentences[i])
        sentences[i] = [w in word_tokenized if w not in stops]
    # Apply lemmatization
    sentences = lemmatize(sentences)
    return sentences

'''
Takes a filename in the form of docA-*.txt
Where A is an integer.
Returns a list of strings, where each string is the body of an article in cluster A.
'''
def extract(filename):
    names = glob.glob(filename)
    articles = []
    for n in names:
        f = open(n, 'r')
        articles.append(f.read().replace('\n',' '))
        f.close()
    return articles

'''
Original implementation of sumbasic from the paper.
Articles is a list(one per article) of lists(one for each sentence) of lists of strings (one for each word).
Returns a string summary.
'''
def orig(articles):
    return True

'''
Simplified sumbasic without non-redundancy update.
'''
def simplified(articles):
    return True

'''
Takes the leading sentences of one of one of the articles(arbitrary selection), up until a word limit is reached.
'''
def leading(articles, word_lim):
    summary = []
    for sentence in articles[0]:
        for word in sentence:
            if len(summary) < word_lim:
                summary.append(word)
            else:
                break
    return summary

'''
Returns the results of calling the method referred to by 'name' with dataset input.
'''
def call_method(name, dataset):
    if name == 'orig':
        return orig(dataset)
    elif name == 'simplified':
        return simplified(dataset)
    elif name == 'leading':
        return leading(dataset, LIMIT)
    else:
        return ["The provided name is not in the list of available methods."]

def pretty_print(sentences):
    for s in sentences:
        if isInstance(s, basestring):
            print(s)
        else:
            print(' '.join(s) + "\n")

def main(method_name, file_n):
    articles = extract(file_n)
    processed = [preprocess(a) for a in articles]
    return call_method(method_name, processed)

if __name__ == "__main__":
    args = sys.argv
    if len(args) != 3:
        print("Wrong argument count.")
    else:
        pretty_print(main(args[1], args[2]))
