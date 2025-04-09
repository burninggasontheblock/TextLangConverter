import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

# Load data
df = pd.read_csv('language.csv')

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text'] = df['text'].apply(clean_text)

# Define features and labels
X = df['text']
y = df['language']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    analyzer='char_wb',         # character n-grams within word boundaries
    ngram_range=(2, 4),         # include bigrams to 4-grams
    max_df=0.9,                 # ignore very common character combos
    min_df=2,                   # ignore very rare ones
    sublinear_tf=True
)

X_vectorized = vectorizer.fit_transform(X)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(df['language'].value_counts())
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
