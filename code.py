import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer
from contractions import fix
from imblearn.over_sampling import RandomOverSampler 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


nltk.download("wordnet")

df = pd.read_csv("C:\\Users\\acer\\Downloads\\Sarcasm.csv")

# --------------data cleaning---------------

df["tweet"].fillna(df["tweet"].mode()[0],inplace=True)
# print(df.isnull().sum())
df = df[["tweet","sarcastic"]]
# print(df.head())

def clean_text(text):
    text = fix(text)
    text = text.lower()
    text = re.sub(r"http\S+", " ", text) 
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word not in stop_words]
    lemmitizer = WordNetLemmatizer()
    stem_words = [lemmitizer.lemmatize(word) for word in filtered_words]
    final_word = " ".join(stem_words)
    return final_word

df["tweet"] = df["tweet"].apply(clean_text)

# --------------vectorization---------------

tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df["tweet"])
y = df["sarcastic"]

# -------------resampling-------------

ros = RandomOverSampler()
X_resample,y_resample = ros.fit_resample(X,y)

# ------------splitting data-----------------

X_train,X_test,y_train,y_test = train_test_split(X_resample,y_resample,test_size=0.2,random_state=42)

# print("Train shape:", X_train.shape)
# print("Test shape:", X_test.shape)
# print("Class distribution after resampling:\n", pd.Series(y_resample).value_counts())

# ------------modeling & evaluation---------------

rf_classifier = RandomForestClassifier()

rf_classifier.fit(X_train,y_train)
y_pred = rf_classifier.predict(X_test)

# print(accuracy_score(y_test,y_pred))
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))

pickle.dump(rf_classifier,open("Day11_nlp\\model\\rf_nlpmodel.pkl","wb"))
pickle.dump(tfidf_vectorizer,open("Day11_nlp\\model\\tfidf_nlpmodel.pkl","wb"))

def detect_sarcasm(text):
    cln_fun = clean_text(text)
    features = tfidf_vectorizer.transform([cln_fun])
    result = rf_classifier.predict(features)
    if result[0] == 1:
        return "Sarcastic"
    else:
        return "Not Sarcastic"

input_text = "Oh great, another Monday! I just love waking up early after the weekend."
print(detect_sarcasm(input_text))

