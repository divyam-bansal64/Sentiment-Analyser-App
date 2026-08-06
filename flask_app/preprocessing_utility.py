import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK datasets quietly if needed
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Pre-instantiate stop words set and lemmatizer ONCE at module load for speed
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def lower_case(text: str) -> str:
    """Converts text to lowercase."""
    return text.lower() if isinstance(text, str) else ""


def remove_html_tags(text: str) -> str:
    """Strips HTML tags like <br /> across single and multiline strings."""
    return re.sub(r'<[^>]+>', ' ', text)


def removing_urls(text: str) -> str:
    """Removes http/https URLs and www domain links."""
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def removing_punctuations(text: str) -> str:
    """Removes punctuation characters, replacing them with whitespace."""
    text = text.replace('؛', '')
    return re.sub(f"[{re.escape(string.punctuation)}]", ' ', text)


def remove_stop_words(text: str) -> str:
    """Filters out English stop words."""
    return " ".join([word for word in text.split() if word not in STOP_WORDS])


def lemmatization(text: str) -> str:
    """Lemmatizes each word in the text to its dictionary base form."""
    return " ".join([LEMMATIZER.lemmatize(word) for word in text.split()])


def preprocess_text(text: str) -> str:
    """
    Applies the full text normalization pipeline while retaining numbers
    for sentiment rating signals (e.g., 10/10, 1 star).
    Matches capstone_src/data/data_preprocessing.py preprocessing logic.
    """
    text = lower_case(text)
    text = remove_html_tags(text)
    text = removing_urls(text)
    text = removing_punctuations(text)
    text = remove_stop_words(text)
    text = lemmatization(text)
    return text
