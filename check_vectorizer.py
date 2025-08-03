import pickle

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print(type(vectorizer))

# If it's a TfidfVectorizer, you're good
if hasattr(vectorizer, 'get_feature_names_out'):
    print("Vocabulary preview:", vectorizer.get_feature_names_out()[:10])
else:
    print("This is NOT a fitted TfidfVectorizer.")
