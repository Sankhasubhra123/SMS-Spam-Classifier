import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training corpus and labels
corpus = [
    "You have won ₹10000! Claim now!",
    "Meeting at 5 PM tomorrow",
    "Congratulations! You've been selected for a prize",
    "Don't forget our call later today"
]
labels = [1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam

# Fit vectorizer and transform
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Save both objects
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Vectorizer and model trained with matching feature dimensions.")
