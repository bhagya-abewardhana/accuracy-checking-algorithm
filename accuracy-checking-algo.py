import numpy
import matplotlib.pyplot
import pandas

dataset = pandas.read_csv('amazonreviews.tsv', delimiter = '\t', quoting = 3)

#print(dataset)

import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
#import contractions # To expand the short form
corpus = []
for i in range(0, 10000):
  review = re.sub('[^a-zA-Z]', ' ', dataset['review'][i]) # Remove all the punctuations
  review = review.lower()
  #review = contractions.fix(review) #To expand the short form
  #review = review.split() # Split it to different elements (words) Tokenization
  review = word_tokenize(review)
  lemmatizer = WordNetLemmatizer()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = [lemmatizer.lemmatize(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)

print(corpus)

# dataset1 = pd.read_json('Review.json', lines = True)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 0].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.21, random_state = 109)



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
classifier = SVC(kernel = 'rbf', gamma= 'scale' , C=1 , random_state = 1)
#cross_validation = StratifiedKFold(n_splits=5,random_state=5,shuffle=True)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(numpy.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)