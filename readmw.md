Personalized Recommender System 

A hybrid recommendation engine that suggests personalized food products to users using both collaborative filtering (SVD) 
and content-based filtering (TF-IDF). Built with `scikit-learn`, `Surprise`, and deployed via `Streamlit`.


Features

- Collaborative filtering (SVD)
- Content-based filtering using TF-IDF over product reviews
- Hybrid recommendations based on both methods
- Search by keyword (e.g., "dog food") instead of exact product IDs
- Interactive Streamlit dashboard

Project Structure

  personalized-recommender/
  - app/
    - dashboard.py
  - data/
    - Reviews.csv(Not included)
  - notebook/
    - Collaborative_Filtering
    - Content_Based
    - eda
    - Hybrid_Model
  - README.md


Dataset

Amazon Fine Food Reviews dataset.
      [Download it from Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

The file is excluded from this repository due to GitHub's 100MB file size limit.

Then place the file : personalized-recommender/
                          - data/
                            - Reviews.csv

How to Run the Project

1.Install the required packages manually
2.Run the Streamlit App:
               streamlit run app/dashboard.py
3.View in your browser

