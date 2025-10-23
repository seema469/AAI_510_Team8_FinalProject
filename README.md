# 🍽️ Recipe Recommendation System & Sentiment Analysis  
**University of San Diego — AAI 510 (Team 8 Final Project)**  

---

## 🧠 Project Overview
This project builds an intelligent **culinary recommendation platform** that predicts user preferences and analyzes recipe reviews.  
It merges **collaborative filtering** and **sentiment analysis (NLP)** to deliver personalized and meaningful recipe suggestions.  

> 🎯 Goal: Enhance user engagement by combining **recommendation systems** with **opinion mining**.

---

## 🎯 Objectives
- Recommend recipes using both user history and ingredient similarity.  
- Perform sentiment analysis on user reviews to understand preferences.  

---

## 🧩 Dataset
**Food.com Dataset** — contains over **500,000 recipes** and **1.4 million user reviews**.

Each entry includes:
- Recipe ID, Name, Ingredients, and Tags  
- User Reviews and Ratings (1–5)  
- Review Text for sentiment analysis  

**Source:** [Kaggle — Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing
- Cleaned and normalized text data (stopword removal, lemmatization).  
- Built user–item matrices for collaborative filtering.  
- Tokenized ingredient lists for semantic similarity modeling.  

### 2️⃣ Recommendation System
**Collaborative Filtering (SVD):**
- Generated latent factors using **Singular Value Decomposition (SVD)**.  
- Calculated **cosine similarity** between user and item vectors.  
- Predicted unseen ratings based on top-N similar items.

**Content-Based Filtering (Word2Vec):**
- Used **Word2Vec embeddings** to capture ingredient similarity.  
- Recommended similar recipes for first-time (cold-start) users.

### 3️⃣ Sentiment Analysis
- Classified review text polarity using traditional and neural models.  
- Models used:  
  - Logistic Regression  
  - Naïve Bayes  
  - Bi-LSTM (for contextual understanding)  
- Evaluated using **Accuracy**, **Precision**, **Recall**, and **F1-score**.

### 4️⃣ Evaluation
- **RMSE** for rating prediction accuracy.  
- **Precision@K / Recall@K** for recommender performance.  
- **F1-score** for sentiment classification.

---

## 🧰 Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| Programming | Python 3.9+ |
| Data Handling | pandas, NumPy |
| NLP | NLTK, spaCy, Word2Vec |
| ML / DL | scikit-learn, TensorFlow / Keras |
| Recommender | Surprise (SVD), Cosine Similarity |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook / Google Colab |

---

**Key Takeaways:**
- The hybrid SVD + Word2Vec system improves personalization.  
- Sentiment-aware filtering enhances overall recommendation quality.  

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/seema469/AAI_510_Team8_FinalProject.git
cd AAI_510_Team8_FinalProject
