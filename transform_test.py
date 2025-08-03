import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import string

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [i for i in tokens if i.isalnum()]
    tokens = [i for i in tokens if i not in stop_words and i not in string.punctuation]
    tokens = [ps.stem(i) for i in tokens]
    return " ".join(tokens)

print(transform_text("Congratulations! You've won a free iPhone 📱. Claim now!!!"))

