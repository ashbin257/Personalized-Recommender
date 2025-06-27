
import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
df = pd.read_csv("C:/Users/ashbi/OneDrive/Desktop/tech/personalized-recommender/data/Reviews.csv")
df = df[['UserId', 'ProductId', 'Score', 'Summary', 'Text']].dropna()
df['combined'] = df['Summary'].astype(str) + " " + df['Text'].astype(str)
df = df.drop_duplicates(subset=['ProductId']).reset_index(drop=True)

# Create TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = tfidf.fit_transform(df['combined'])

# Content-based model
content_model = NearestNeighbors(metric='cosine', algorithm='brute')
content_model.fit(tfidf_matrix)

# Collaborative model
reader = Reader(rating_scale=(1, 5))
cf_data = Dataset.load_from_df(df[['UserId', 'ProductId', 'Score']], reader)
trainset = cf_data.build_full_trainset()
cf_model = SVD()
cf_model.fit(trainset)

# Product index mapping
product_indices = pd.Series(df.index, index=df['ProductId'])

# Recommendation function
def hybrid_recommend(user_id, product_id, top_n=5):
    if product_id not in product_indices:
        return pd.DataFrame([["Product not found", "", None]], columns=["ProductId", "Summary", "PredictedRating"])

    idx = product_indices[product_id]
    distances, indices = content_model.kneighbors(tfidf_matrix[idx], n_neighbors=top_n+1)
    similar_indices = indices.flatten()[1:]

    results = []
    for i in similar_indices:
        pid = df.iloc[i]['ProductId']
        try:
            pred = cf_model.predict(str(user_id), str(pid))
            est_rating = pred.est
        except:
            est_rating = None
        results.append((pid, df.iloc[i]['Summary'], est_rating))

    results = sorted(results, key=lambda x: x[2] if x[2] is not None else 0, reverse=True)
    return pd.DataFrame(results, columns=["ProductId", "Summary", "PredictedRating"])

# Streamlit UI
st.title("🍽️ Hybrid Food Recommendation System")

user_input = st.text_input("Enter User ID", "")
product_input = ""
search_text = st.text_input("🔎 Search for a product (e.g. 'dog food')")

if search_text:
    search_vec = tfidf.transform([search_text])
    distances, indices = content_model.kneighbors(search_vec, n_neighbors=1)
    matched_product = df.iloc[indices[0][0]]['ProductId']
    matched_summary = df.iloc[indices[0][0]]['Summary']
    st.markdown(f"✅ Closest match: **{matched_summary}** (`{matched_product}`)")
    product_input = matched_product
else:
    product_input = st.text_input("Enter Product ID (optional if searched above)", "")
top_n = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Get Recommendations"):
    if user_input and product_input:
        results = hybrid_recommend(user_input, product_input, top_n)
        st.write("Top Recommendations:")
        st.dataframe(results)
    else:
        st.warning("Please enter both User ID and Product ID.")
