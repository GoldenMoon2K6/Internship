# Import necessary libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC  # Changed to LinearSVC for better performance
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import nltk

# Set NLTK data path to use the local NLTK data (offline mode)
nltk.data.path.append('E:/Python/nltk_data')
nltk.download('stopwords', force=True)
nltk.download('punkt', force=True, download_dir='E:/Python/nltk_data')

# File paths for the datasets
train_data_file = 'E:/Internship/internship hemachandra/Movie_Genre_Classification/Genre Classification Dataset/train_data.txt'
test_data_file = 'E:/Internship/internship hemachandra/Movie_Genre_Classification/Genre Classification Dataset/test_data.txt'
solution_file = 'E:/Internship/internship hemachandra/Movie_Genre_Classification/Genre Classification Dataset/test_data_solution.txt'
output_file = 'E:/Internship/internship hemachandra/Movie_Genre_Classification/test_data_with_predictions.txt'

# Load the training data with the correct delimiter
train_data = pd.read_csv(train_data_file, delimiter=' ::: ', engine='python', header=None)
train_data.columns = ['movie_id', 'title', 'genre', 'plot_summary']

# Preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

# Preprocess plot summaries
train_data['cleaned_plot'] = train_data['plot_summary'].apply(preprocess_text)

# Encode target labels (genres)
label_encoder = LabelEncoder()
train_data['encoded_genre'] = label_encoder.fit_transform(train_data['genre'])

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(train_data['cleaned_plot'])  # Features (plot summaries)
y = train_data['encoded_genre']  # Labels (encoded genres)

# Train the models on the entire dataset
models = {
    'Logistic Regression': LogisticRegression(max_iter=200, verbose=1, solver='saga'),
    'SVM': LinearSVC(),  # Changed to LinearSVC for efficiency
    'Naive Bayes': MultinomialNB()
}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X, y)  # Use entire dataset
    y_pred = model.predict(X)

    # Evaluation
    print(f"Accuracy of {model_name}: {accuracy_score(y, y_pred)}")
    print(f"Classification Report for {model_name}:\n{classification_report(y, y_pred, zero_division=0)}\n")

# Save the best performing model, vectorizer, and label encoder
best_model = LogisticRegression(max_iter=200, solver='saga')
best_model.fit(X, y)

joblib.dump(best_model, 'E:/Internship/internship hemachandra/Movie_Genre_Classification/movie_genre_classifier.pkl')
joblib.dump(vectorizer, 'E:/Internship/internship hemachandra/Movie_Genre_Classification/tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'E:/Internship/internship hemachandra/Movie_Genre_Classification/label_encoder.pkl')

# Load and preprocess the test data
test_data = pd.read_csv(test_data_file, delimiter=' ::: ', engine='python', header=None)
test_data.columns = ['movie_id', 'title', 'plot_summary']

# Preprocess plot summaries in the test data
test_data['cleaned_plot'] = test_data['plot_summary'].apply(preprocess_text)

# Transform the test data using the trained vectorizer
X_test = vectorizer.transform(test_data['cleaned_plot'])

# Make predictions on the test data
predicted_genres_encoded = best_model.predict(X_test)

# Convert encoded genres back to original labels
predicted_genres = label_encoder.inverse_transform(predicted_genres_encoded)

# Add predictions to test data
test_data['predicted_genre'] = predicted_genres

# Save predictions in the required text format
with open(output_file, 'w', encoding='utf-8') as f:
    for index, row in test_data.iterrows():
        f.write(f"{row['movie_id']} ::: {row['title']} ::: {row['predicted_genre']} ::: {row['plot_summary']}\n")

print(f"Predictions on test data saved to '{output_file}'.")
