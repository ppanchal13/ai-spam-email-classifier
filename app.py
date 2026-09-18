from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model and TF-IDF vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    message = ""

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if message:
            # Convert message into TF-IDF features
            message_vector = vectorizer.transform([message])

            # Make prediction
            result = model.predict(message_vector)[0]

            # Get prediction probability
            probabilities = model.predict_proba(message_vector)[0]
            confidence = round(max(probabilities) * 100, 2)

            if result == 1:
                prediction = "Spam"
            else:
                prediction = "Not Spam"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        message=message
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
