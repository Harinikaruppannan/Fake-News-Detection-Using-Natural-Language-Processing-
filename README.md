### FAKE NEWS DETECTION USING(NLP)

### Packages imported :

- NLTK and Scikit-learn: Essential for natural language processing and machine learning tasks.
- TensorFlow or PyTorch: Deep learning frameworks for neural network implementation.
- Pandas: Crucial for data manipulation and cleaning.
- Word Embeddings (Word2Vec or GloVe): Enhances semantic understanding in language.
- Flask or Django: Web frameworks for deploying the model as a web application.
- Beautiful Soup: Useful for web scraping if collecting data from online sources.
- Matplotlib or Seaborn: Data visualization tools for insights presentation.

ALGORITHM for Data Preprocessing :

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the true and fake news datasets
true_df = pd.read_csv('true.csv')
fake_df = pd.read_csv('fake.csv')

# Add a column to indicate the type of news (true or fake)
true_df['NewsType'] = 'True'
fake_df['NewsType'] = 'Fake'

print(fake_df,true_df)

# Concatenate both datasets
combined_df = pd.concat([true_df, fake_df], ignore_index=True)

# Filter and display only True news
true_news = combined_df[combined_df['NewsType'] == 'True']
print("True News:")
print(true_news.head())

# Filter and display only Fake news
fake_news = combined_df[combined_df['NewsType'] == 'Fake']
print("Fake News:")
print(fake_news.head())

# Visualize the distribution of news types
news_type_counts = combined_df['NewsType'].value_counts()
news_type_counts.plot(kind='bar', color=['green', 'red'])
plt.title('Distribution of News Types')
plt.xlabel('News Type')
plt.ylabel('Count')
plt.show()

ALGORITHM for Model Training :

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# Load the true and false datasets
true_data = pd.read_excel("true.xlsx")
false_data = pd.read_excel("false.xlsx")

# Combine the datasets into one
true_data['label'] = 1
false_data['label'] = 0
data = pd.concat([true_data, false_data], ignore_index=True)

# Text Preprocessing
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

data['text'] = data['text'].str.lower()
data['text'] = data['text'].apply(preprocess_text)

# Feature Extraction: TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features as needed
X = tfidf_vectorizer.fit_transform(data['text'])
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training: Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Model Evaluation
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy and print classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# Create a simple accuracy plot
plt.bar(['Accuracy'], [accuracy], color='blue')
plt.ylim(0, 1)  # Set the y-axis limit to show accuracy between 0 and 1
plt.ylabel('Accuracy')
plt.title('Fake News Detection Accuracy')
plt.show()
