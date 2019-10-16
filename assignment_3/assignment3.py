import random
import codecs
import string
from nltk.stem.porter import PorterStemmer
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel
from gensim.similarities import MatrixSimilarity
from progressbar import printProgressBar

random.seed(123)


# Part 1: Data loading and pre-processing
def process_text(text_file):
    """
    This function takes an input text file, removes unecessary headers and punctuation and whitespaces, and splits it
    into paragraphs. It then stems the tokenized words in each paragraph.

    :param text_file: Name of text file to tokenize and process.
    :return: List of lists. Each list represents a paragraph of stemmed words.
    """
    stemmer = PorterStemmer()

    with codecs.open(text_file, "r", "utf-8") as f:
        paragraphs = f.read().split(
            '\n\n')  # Reads the text file and converts to lower case, removes all punctuation and line breaks

    # Remove paragraphs containing the word "Guthenberg"
    paragraphs_without_header_and_footer = []
    for paragraph in paragraphs:
        if paragraph != "":
            if "gutenberg" not in paragraph.lower():
                paragraphs_without_header_and_footer.append(paragraph)

    # Tokenize the list (split the strings into single words)
    tokenize = [None for i in range(len(paragraphs_without_header_and_footer))]
    for i in range(len(paragraphs_without_header_and_footer)):
        tokenize[i] = paragraphs_without_header_and_footer[i].lower().translate(
            str.maketrans('', '', string.punctuation)).split()

    printProgressBar(0, len(tokenize), prefix='Pre-processing text:', suffix='Complete',
                     length=50)  # Prints progress bar
    # Stem all the words in tokenize
    for i in range(len(tokenize)):
        printProgressBar(i + 1, len(tokenize), prefix='Pre-processing text:', suffix='Complete', length=50)
        tokenize[i] = [stemmer.stem(word) for word in tokenize[i]]

    return tokenize, paragraphs_without_header_and_footer


# Part 2: Dictionary building
def build_dictionary(text_file, stop_words):
    """
    This function takes a text file and a file of stop words, and builds dictionary with pairs of word indexes and word
    counts in every paragraph.

    :param text_file: Input text file
    :param stop_words: Text file of stop words
    :return: Corpus object (=list of paragraphs); each paragraph is a list of pairs (word-index, word-count)
    """
    words, paragraphs = process_text(text_file)
    dictionary = Dictionary(words)

    # Gather all stop words
    with codecs.open(stop_words, "r", "utf-8") as stop_w:
        stop_words = stop_w.read().split(',')

    # Gather all stop word ids
    stop_word_ids = []
    for i in range(len(dictionary)):
        if dictionary[i] in stop_words:  # Check if stop word exists in dictionary
            stop_word_ids.append(dictionary.token2id[dictionary[i]])
    dictionary.filter_tokens(stop_word_ids)  # Filter out all stop words

    bags_of_words = []
    printProgressBar(0, len(words), prefix='Building dictionary:', suffix='Complete', length=50)
    for i in range(len(words)):
        printProgressBar(i + 1, len(words), prefix='Building dictionary:', suffix='Complete', length=50)
        bags_of_words.append(dictionary.doc2bow(words[i]))

    return bags_of_words, dictionary, paragraphs


# Part 3: Retrieval Models
def tf_idf(corpus):
    tfidf_model = TfidfModel(corpus)
    tfidf_corpus = []
    for i in range(len(corpus)):
        tfidf_corpus.append(tfidf_model[corpus[i]])

    tfidf_similarity_matrix = MatrixSimilarity(tfidf_corpus)

    return tfidf_similarity_matrix


def lsi(corpus, dictionary):
    lsi_model = LsiModel(corpus, id2word=dictionary, num_topics=100)
    lsi_corpus = []
    for i in range(len(corpus)):
        lsi_corpus.append(lsi_model[corpus[i]])

    lsi_similarity_matrix = MatrixSimilarity(lsi_corpus)
    print(lsi_model.show_topics())
    return lsi_similarity_matrix


# Part 4: Querying
def pre_processing(query):
    stemmer = PorterStemmer()
    tokenize = query.lower().translate(str.maketrans('', '', string.punctuation)).split(' ')
    tokenize_stemmed = [stemmer.stem(word) for word in tokenize]
    return tokenize_stemmed


def process_query(query, dictionary):
    query_processed = pre_processing(query)
    query_processed = dictionary.doc2bow(query_processed)
    return query_processed


def print_query_weight_result(query_tfidf_weights):
    for i in range(len(query_tfidf_weights)):
        print(dictionary[query_tfidf_weights[i][0]], ":", round(query_tfidf_weights[i][1], 2))


def custom_queries(corpus, dictionary, paragraphs):

    # tfidf query:
    tfidf_model = TfidfModel(corpus, dictionary=dictionary)
    query = process_query("What is the function of money?", dictionary)
    tfidf_query = tfidf_model[query]

    tfidf_corpus = []
    for i in range(len(corpus)):
        tfidf_corpus.append(tfidf_model[corpus[i]])

    tfidf_index = MatrixSimilarity(tfidf_corpus)

    print("tfidf query:")
    doc2similarity_tfidf = enumerate(tfidf_index[tfidf_query])
    for tfidf_index, similarity in sorted(doc2similarity_tfidf, key=lambda kv: -kv[1])[:3]:
        paragraph = paragraphs[tfidf_index].split("\n")
        number = tfidf_index + 1
        print("[paragraph: " + str(number) + "]")
        for i in range(5):
            print(paragraph[i])
            if (i+1) == len(paragraph):
                break
        print("\n")

    # lsi query:
    lsi_model = LsiModel(corpus, id2word=dictionary, num_topics=100)
    lsi_query = lsi_model[tfidf_query]

    lsi_corpus = []
    for i in range(len(corpus)):
        lsi_corpus.append(lsi_model[corpus[i]])

    lsi_index = MatrixSimilarity(lsi_corpus)
    doc2similarity_lsi = enumerate(lsi_index[lsi_query])

    print("lsi query:")
    for lsi_index, similarity in sorted(doc2similarity_lsi, key=lambda kv: -kv[1])[:3]:
        paragraph = paragraphs[lsi_index].split("\n")
        number = lsi_index + 1
        print("[paragraph: " + str(number) + "]")
        for i in range(5):
            print(paragraph[i])
            if (i + 1) == len(paragraph):
                break
        print("\n")


if __name__ == "__main__":
    corpus, dictionary, paragraphs = build_dictionary("pg3300.txt", "stopwords.txt")
    # lsi(corpus, dictionary)

    custom_queries(corpus, dictionary, paragraphs)
