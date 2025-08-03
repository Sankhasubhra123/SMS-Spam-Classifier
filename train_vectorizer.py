from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

corpus = [
    "You have won ₹10000! Claim now!",
    "Meeting at 5 PM tomorrow",
    "Congratulations! You've been selected for a prize",
    "Don't forget our call later today"
]

tfidf = TfidfVectorizer()
tfidf.fit(corpus)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✅ Fitted TfidfVectorizer saved successfully.")
