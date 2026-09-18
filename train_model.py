import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib


# Load dataset
data = pd.read_csv(
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("Dataset loaded successfully!")
print("Total messages:", len(data))


# Convert labels into numbers
# ham = 0 (not spam)
# spam = 1
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data["message"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)


# Convert text into numerical features
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Train model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)


# Test model
predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Evaluation")
print("----------------")
print(f"Accuracy: {accuracy:.4f}") 

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Not Spam", "Spam"]
    )
)


# Save trained model and TF-IDF vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nModel saved as spam_model.pkl")
print("Vectorizer saved as tfidf_vectorizer.pkl")
