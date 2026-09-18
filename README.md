# 📧 AI Spam Email Classifier

A machine-learning based text classification application that detects whether a message is **Spam** or **Not Spam**.

The project uses **TF-IDF vectorization** to convert text into numerical features and a **Logistic Regression** classifier to make predictions. A Flask web interface allows users to enter a message and receive a classification along with the model's confidence score.

---

## 🚀 Features

- Spam / Not Spam classification
- TF-IDF based text feature extraction
- Logistic Regression classification model
- Prediction confidence score
- Interactive Flask web interface
- Separate model training pipeline
- Saved trained model for inference
- Model evaluation using precision, recall and F1-score

---

## 🧠 How It Works

```text
User Message
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Logistic Regression Model
     ↓
Spam / Not Spam Prediction
     ↓
Prediction Confidence
```

### 1. TF-IDF Vectorization

Machine-learning models cannot directly process raw text.

TF-IDF converts the message into numerical features by measuring how informative different words are across the training data.

### 2. Logistic Regression

The numerical TF-IDF features are passed to a Logistic Regression classifier trained to distinguish between spam and legitimate messages.

### 3. Prediction

For a new message, the application:

1. Transforms the text using the trained TF-IDF vectorizer
2. Passes the resulting features to the trained model
3. Predicts Spam or Not Spam
4. Displays the model's confidence score

---

## 📊 Model Performance

The model was evaluated on a held-out 20% test set.

| Metric | Result |
|---|---:|
| Accuracy | **97.67%** |
| Spam Precision | **91%** |
| Spam Recall | **91%** |
| Spam F1-Score | **91%** |

The test set contained **1,115 messages**, including **149 spam messages**.

---

## 📚 Dataset

The model is trained using the **SMS Spam Collection** dataset containing:

**5,572 labeled messages**

The messages are classified into:

- `ham` → legitimate / non-spam
- `spam` → spam

> Although the application is presented as a spam message classifier, the current training dataset consists primarily of SMS-style messages rather than a full email corpus.

---

## 🛠️ Tech Stack

- **Python**
- **Flask**
- **Scikit-learn**
- **Pandas**
- **TF-IDF**
- **Logistic Regression**
- **Joblib**
- **HTML / CSS**

---

## 📁 Project Structure

```text
ai-spam-email-classifier/
│
├── app.py
├── train_model.py
├── spam_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── README.md
│
└── templates/
    └── index.html
```

### `train_model.py`

Loads the dataset, creates the train/test split, performs TF-IDF vectorization, trains the Logistic Regression model, evaluates its performance and saves the trained artifacts.

### `app.py`

Loads the trained model and vectorizer and serves predictions through the Flask application.

### `templates/index.html`

Provides the user interface for entering messages and viewing classification results.

---

## 💻 Run Locally

Clone the repository:

```bash
git clone https://github.com/ppanchal13/ai-spam-email-classifier.git
```

Move into the project directory:

```bash
cd ai-spam-email-classifier
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

in your browser.

---

## 🧪 Train the Model

To retrain the classifier:

```bash
python train_model.py
```

This generates:

```text
spam_model.pkl
tfidf_vectorizer.pkl
```

which are then loaded by the Flask application for inference.

---

## 🔍 Example

### Input

```text
Congratulations! You have won a FREE cash prize.
Click now to claim your reward!
```

### Output

```text
🚨 Spam Detected
Model Confidence: 98.02%
```

The application can also classify normal conversational messages as **Not Spam**.

---

## ⚠️ Current Limitations

- The current model is trained on SMS-style text rather than a large modern email corpus.
- TF-IDF does not deeply understand semantic meaning or context.
- Confidence values represent model probabilities and should not be interpreted as guaranteed correctness.
- Real-world phishing and spam can use more sophisticated language than the training dataset contains.
- The application currently analyzes message text only and does not inspect URLs, attachments, sender reputation or email metadata.

---

## 🔮 Future Improvements

Possible future extensions include:

- Training on a larger email-specific dataset
- Transformer-based text classification
- URL and phishing-link analysis
- Sender and email metadata analysis
- Explainable predictions showing influential words
- REST API endpoint for external applications
- Docker containerization
- Cloud deployment
- Automated testing and CI/CD

---

## 👨‍💻 Author

**Parth Panchal**

Computer Engineering Student — NMIMS, Navi Mumbai  
Class of 2027

[GitHub](https://github.com/ppanchal13) · [LinkedIn](https://www.linkedin.com/in/parth-panchal-914b3141a)
