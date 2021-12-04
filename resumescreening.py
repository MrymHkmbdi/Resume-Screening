import pandas as pd

pip install clean-text
pip install nltk

import nltk
nltk.download('punkt')
nltk.download('stopwords')

import re
import nltk
import string
import pandas as pd
from sklearn import metrics
from cleantext import clean
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

resumes = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/resume/UpdatedResumeDataSet.csv', encoding='utf-8')
resumes['cleaned_resume'] = ''

resumes.head()

def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText
    
resumes['cleaned_resume'] = resumes.Resume.apply(lambda x: cleanResume(x))


text = resumes['cleaned_resume'].values
target = resumes['Category'].values

word_vec = TfidfVectorizer(max_features=1500).fit_transform(text)

X_train, X_test, y_train, y_test = train_test_split(word_vec, target, random_state=0, test_size=0.2)

classifier = OneVsRestClassifier(KNeighborsClassifier()).fit(X_train, y_train)
prediction = classifier.predict(X_test)
print('Accuracy: {:.2f}'.format(classifier.score(X_test, y_test)))
