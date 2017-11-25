from nltk.stem.wordnet import WordNetLemmatizer
import sys

'''
Takes in a string (data), and applies standard preprocessing to it, returning a list of tokens.
'''
def preprocess(data):
    return list(data)

'''
Takes a filename in the form of docA-*.txt
Where A is an integer.
Returns a list of strings, where each string is the body of an article in cluster A.
'''
def extract(filename):
    return filename

'''
Original implementation of sumbasic from the paper.
Articles is a list of lists of string tokens.
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
    return True

'''
Returns the results of calling the method referred to by 'name' with dataset input.
'''
def call_method(name, dataset):
    if name == 'orig':
        return orig(dataset)
    elif name == 'simplified':
        return simplified(dataset)
    elif name == 'leading':
        return leading(dataset)
    else:
        return "The provided name is not in the list of available methods."

def main(method_name, file_n):
    articles = extract(file_n)
    processed = [preprocess(a) for a in articles]
    return call_method(method_name, processed)

if __name__ == "__main__":
    args = sys.argv
    if len(args) != 3:
        print("Wrong argument count.")
    else:
        print(main(args[1], args[2]))
