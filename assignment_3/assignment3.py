import random
import codecs
import string
from nltk.stem.porter import PorterStemmer
from gensim.corpora import Dictionary
random.seed(123)


def process_text(text_file):
    """
    This function takes an input text file, removes unecessary headers and punctuation and whitespaces, and splits it
    into paragraphs. It then stems the tokenized words in each paragraph.

    :param text_file: Name of text file to tokenize and process.
    :return: List of lists. Each list represents a paragraph of stemmed words.
    """
    stemmer = PorterStemmer()

    with codecs.open(text_file, "r", "utf-8") as f:
        paragraphs = f.read().lower().translate(str.maketrans('', '', string.punctuation)).split(
            '\n\n')  # Reads the text file and converts to lower case, removes all punctuation and line breaks

    # Remove paragraphs containing the word "Guthenberg"
    paragraphs_without_header_and_footer = []
    for paragraph in paragraphs:
        if paragraph != "":
            if "gutenberg" not in paragraph:
                paragraphs_without_header_and_footer.append(paragraph)

    # Tokenize the list (split the strings into single words)
    tokenize = [None for i in range(len(paragraphs_without_header_and_footer))]
    for i in range(len(paragraphs_without_header_and_footer)):
        tokenize[i] = paragraphs_without_header_and_footer[i].split()

    # Stem all the words in tokenize
    for i in range(len(tokenize)):
        tokenize[i] = [stemmer.stem(word) for word in tokenize[i]]

    return tokenize


def build_dictionary(text_file, stop_words):
    """
    This function takes a text file and a file of stop words, and builds dictionary with pairs of word indexes and word
    counts in every paragraph.

    :param text_file: Input text file
    :param stop_words: Text file of stop words
    :return: Corpus object (=list of paragraphs); each paragraph is a list of pairs (word-index, word-count)
    """
    words = process_text(text_file)
    dictionary = Dictionary(words)

    # Gather all stop words
    with codecs.open(stop_words, "r", "utf-8") as stop_w:
        stop_words = stop_w.read().split(',')

    # Gather all stop word ids
    stop_word_ids = []
    for i in range(len(dictionary)):
        if dictionary[i] in stop_words:     # Check if stop word exists in dictionary
            stop_word_ids.append(dictionary.token2id[dictionary[i]])
    dictionary.filter_tokens(stop_word_ids)     # Filter out all stop words

    bags_of_words = []
    for paragraph in words:
        bags_of_words.append(dictionary.doc2bow(paragraph))

    return bags_of_words


if __name__ == "__main__":
    build_dictionary("text.txt", "stopwords.txt")
