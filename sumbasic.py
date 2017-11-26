from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stops = stopwords.words('english')
import sys
import glob
import string
import codecs

wnl = WordNetLemmatizer()
LIMIT = 150
punctuation = string.punctuation

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
    unproc = sentences[:]
    # Apply tokenization within the sentences and store seperately, removing stop words and lowercasing all
    for i in range(len(sentences)):
        sentences[i] = sentences[i].lower()
        word_tokenized = word_tokenize(sentences[i])
        sentences[i] = [w for w in word_tokenized if w not in stops]
    # Apply lemmatization
    sentences = lemmatize(sentences)
    return (sentences, unproc)

'''
Takes a filename in the form of docA-*.txt
Where A is an integer.
Returns a list of strings, where each string is the body of an article in cluster A.
'''
def extract(filename):
    names = glob.glob(filename)
    articles = []
    for n in names:
        f = codecs.open(n, 'r', 'utf-8')
        articles.append(f.read().replace('\n',' ').replace(u"\u2013","-").replace(u"\u2018", "'").replace(u"\u2019", "'").replace(u"\u201C","\"").replace(u"\u201D", "\""))
        f.close()
    return articles

'''
Takes the processed articles
Returns a dictionary mapping word type to P(w) = (# of w tokens)/(# of all tokens)
'''
def word_probs(articles):
    words = dict()
    total = 0
    for s in articles:
        for w in s:
            if w not in punctuation:
                proc = w.translate(punctuation)
                if proc not in words:
                    words[proc] = 1
                else:
                    words[proc] += 1
                total += 1
    for k, v in words.items():
        words[k] = float(v)/total
    return words

'''
Takes the articles and the word probabilities
Returns the weighted sentences in the form of a dictionary mapping sentence to weight
'''
def sent_weights(articles, word_ps):
    weights = dict()
    for s in articles:
        weights[tuple(s)] = sum([word_ps[w.translate(punctuation)] for w in s if w not in punctuation])/float(len(s))
    return weights

'''
Takes the chosen sentence, and word probabilities
Updates the probabilities of the words in the chosen sentence
'''
def update_probs(chosen_sentence, word_ps):
    for w in chosen_sentence:
        if w not in punctuation:
            proc = w.translate(punctuation)
            word_ps[proc] = word_ps[proc]*word_ps[proc]

'''
Returns the best scoring sentence from the set of sentences which contain the highest probability word
'''
def get_best_sent(word_ps, sent_ws):
    # Get highest prob word
    max_so_far = (" ", 0.0)
    for k, v in word_ps.items():
        if v > max_so_far[1]:
            max_so_far = (k, v)
    best_word = max_so_far[0]
    # Find all sentences that have the best word in them
    considered = []
    for k, v in sent_ws.items():
        if best_word in [w.translate(punctuation) for w in k]:
            considered.append((k, v))
    # Find the highest scoring sentence
    max_so_far = ([], 0.0)
    for k, v in considered:
        if v > max_so_far[1]:
            max_so_far = (k, v)
    return max_so_far[0]

'''
Original implementation of sumbasic from the paper.
Articles is a list(one per article) of lists(one for each sentence) of lists of strings (one for each word).
Returns a string summary.
'''
def orig(articles, word_limit):
    processed = []
    unproc = []
    for a in articles:
        for sent1 in a[0]:
            processed.append(sent1)
        for sent2 in a[1]:
            unproc.append(sent2)
    # Summary is a list of sentences. The sentences are lists of strings(words).
    summary = []
    # Calculate word probabilities
    word_probabilities = word_probs(processed)
    while len([w for s in summary for w in s]) < word_limit:
        # Calculate sentence weights
        sentence_weights = sent_weights(processed, word_probabilities)
        # Out of the set of sentences which contain the highest probability word, pick best scoring sentence
        best = get_best_sent(word_probabilities, sentence_weights)
        # Add chosen sentence to summary
        summary.append(unproc[processed.index(list(best))])
        # Update probabilities of all words in chosen sentence
        update_probs(best, word_probabilities)
        # Repeat from 2 if length is lower than desired
    for i in range(len(summary)):
        summary[i] = ''.join(summary[i])
    return '\n'.join(summary)

'''
Simplified sumbasic without non-redundancy update.
'''
def simplified(articles, word_limit):
    processed = []
    unproc = []
    for a in articles:
        for sent1 in a[0]:
            processed.append(sent1)
        for sent2 in a[1]:
            unproc.append(sent2)
    # Summary is a list of sentences. The sentences are lists of strings(words).
    summary = []
    # Calculate word probabilities
    word_probabilities = word_probs(processed)
    while len([w for s in summary for w in s]) < word_limit:
        # Calculate sentence weights
        sentence_weights = sent_weights(processed, word_probabilities)
        # Out of the set of sentences which contain the highest probability word, pick best scoring sentence
        best = get_best_sent(word_probabilities, sentence_weights)
        # Add chosen sentence to summary
        summary.append(unproc[processed.index(list(best))])
        # Repeat from 2 if length is lower than desired
    for i in range(len(summary)):
        summary[i] = ''.join(summary[i])
    return '\n'.join(summary)

'''
Takes the leading sentences of one of one of the articles(arbitrary selection), up until a word limit is reached.
'''
def leading(articles, word_lim):
    summary = []
    for a in articles:
        for sentence in a[1]:
            for word in word_tokenize(sentence):
                if len(summary) < word_lim:
                    summary.append(codecs.encode(word, 'utf-8'))
                else:
                    break
    return ' '.join(summary)

'''
Returns the results of calling the method referred to by 'name' with dataset input.
'''
def call_method(name, dataset):
    if name == 'orig':
        return orig(dataset, LIMIT)
    elif name == 'simplified':
        return simplified(dataset, LIMIT)
    elif name == 'leading':
        return leading(dataset, LIMIT)
    else:
        return "The provided name is not in the list of available methods."

def main(method_name, file_n):
    articles = extract(file_n)
    processed = [preprocess(a) for a in articles]
    #processed = [preprocess(a) for a in articles]
    return call_method(method_name, processed)

if __name__ == "__main__":
    args = sys.argv
    print(main(args[1], args[2]))
