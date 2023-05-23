import re, unicodedata
from nltk.stem import WordNetLemmatizer, PorterStemmer
from stopwords_collection import stopwords

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def strip_accents(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    try:
        text = str(text)
    except (TypeError, NameError):  # unicode is a default on python 3
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    text = text.strip()

    return str(text)


def remove_emails(text):
    """
    removes emails from given sentence
    """
    email_re_pattern = "^[_a-z0-9-]+(\.[_a-z0-9-]+)*@[a-z0-9-]+(\.[a-z0-9-]+)*(\.[a-z]{2,4})$"
    text = [re.sub(email_re_pattern, " ", t) for t in text.split(" ")]
    return " ".join(text).strip()


def remove_mentions_urls(text):
    """
    removes emails, hashtag, user mentions, urls
    """
    text = remove_emails(text)
    # return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split()) # removes punctuation as well!
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|(\w+:\/\/\S+)", " ", text).split())


def remove_placeholders(text):
    """
    removes XX/XX/XXXX, XXXX from the given text
    :param text:
    :return:
    """
    text = text.replace("XXXX", "")
    text = text.replace("XX/XX/XXXX", "")
    text = text.replace("XX/XX/", "")
    return text


def preprocess(document, remove_punct=True, lemmatize=False, stemming=False, lower=True):
    """
    removes email, url and perform lemmatize or stemming, lowers and remove the punctuations
    :param document:
    :param remove_punct:
    :param lemmatize:
    :param lower:
    :return:
    """
    document = remove_placeholders(document)
    document = strip_accents(document)
    document = remove_emails(document)
    document = remove_mentions_urls(document)

    if remove_punct:
        document = re.sub(r'[^\w\s]', '', document)  # # remove punctuations
    if lemmatize:
        document = [w for w in document.lower().split(' ') if w not in stopwords]
        document = " ".join([lemmatizer.lemmatize(w) for w in document if w != ''])
    if stemming:
        document = [w for w in document.lower().split(' ') if w not in stopwords]
        document = " ".join([stemmer.stem(w) for w in document if w != ''])
    if lower:
        document = document.lower()

    return document
