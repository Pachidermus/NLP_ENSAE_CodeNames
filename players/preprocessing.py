from nltk.corpus import stopwords 
from gensim.parsing.preprocessing import STOPWORDS    
import string

STOPWORDS_NLTK = list(stopwords.words('english'))
STOPWORD_NLTK_GENSIM = list(STOPWORDS)
STOPWORD_NLTK_GENSIM.extend(STOPWORDS_NLTK)

# words to keep : gensim removes too many tokens
keep_words = ['find', 'last', 'nine', 'front', 'hundred', 'off', 'few', 'became', 'various', 'fifteen', 'computer', 'forty', 'interest', 'everyone', 'give', 'become', 'keep', 'using', 'serious', 'becomes', 'might', 'above', 'take', 'thick', 'eleven', 'name', 'fifty', 'thin', 'show', 'down', 'move', 'see', 'sincere', 'call', 'becoming', 'bottom', 'top', 'bill', 'under', 'go', 'km', 'eight', 'below', 'cry', 'empty', 'twelve', 'well', 'twenty', 'sixty', 'mill', 'five', 'ten', 'third', 'side', 'detail', 'full', 'everything', 'first', 'fill', 'four', 'six', 'system', 'will', 'our', 'two', 'amount', 'due', 'made', 'fire', 'three', 'part', 'back', 'mine']
for i in keep_words :
    STOPWORD_NLTK_GENSIM.remove(i)

# remove tokens which contain punctuation (balises html, etc.)
punc = string.punctuation

def clean_vocab(model) :
    """function that removes words with punctuations and stopwords from an embedding vocabulary"""
    vocab_set = set(model.vocab.keys())
    drop_set = STOPWORD_NLTK_GENSIM
    for word in list(model.vocab):
        if (word.lower() in drop_set) | (bool([ele for ele in list(punc) if(ele in word)]) == True) :
            del model.vocab[word]