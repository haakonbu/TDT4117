import random
import codecs
import string
from nltk.stem.porter import PorterStemmer

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


if __name__ == "__main__":
    tokenize = process_text("pg3300.txt")
