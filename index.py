import numpy as np

str1 = 'the chosen software developer will be part of a larger engineering team developing software for medical devices.'
str2 = 'we are seeking a seasoned software developer with strong analytical and technical skills to join our public sector technology consulting team.'

import nltk

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# Tokenize the str1 and str2
tokened_str1 = word_tokenize(str1)
tokened_str2 = word_tokenize(str2)

tokened_str1 = [w for w in tokened_str1 if len(w) > 2]
tokened_str2 = [w for w in tokened_str2 if len(w) > 2]

# Remove unecessary word(stopwords)
no_sw_str1 = [word for word in tokened_str1 if not word in stopwords.words()]
no_sw_str2 = [word for word in tokened_str2 if not word in stopwords.words()]

ps = PorterStemmer()
# Select the unique
stemmed_str1 = np.unique([ps.stem(w) for w in no_sw_str1])
stemmed_str2 = np.unique([ps.stem(w) for w in no_sw_str2])

full_list = np.sort(np.concatenate([stemmed_str1, stemmed_str2]))
# Listed all keywords
print(full_list)
