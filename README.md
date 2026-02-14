# Twitter Sentiment Analysis using Sentiment140

This project focuses on performing **sentiment analysis on Twitter data** using Natural Language Processing (NLP) techniques.  
The task is to classify tweets as **positive** or **negative** based on their textual content.

The project includes data exploration, preprocessing, baseline model training, and deployment as a web application.

---

## Project Link

- **Kaggle Notebook (Model Training & EDA):**  
  https://www.kaggle.com/code/leyuzakoksoken/twitter-sentiment-analysis-using-sentiment140

---

## Project Objective

The objective of this project is to build a sentiment classification system that analyzes Twitter texts and predicts whether the sentiment expressed in a tweet is positive or negative.  
The project aims to demonstrate the full NLP pipeline, from exploratory data analysis to model deployment.

---

## Dataset

- **Name:** Sentiment140
- **Size:** 1.6 million tweets
- **Source:** Twitter
- **Labels:**
  - `0` → Negative
  - `1` → Positive

The dataset is perfectly balanced, containing an equal number of positive and negative samples.

---

## Exploratory Data Analysis (EDA)

During the EDA stage:
- Dataset size and structure were examined
- Sentiment label distribution was analyzed
- Tweet length distributions (character and word level) were visualized

Key observations:
- Most tweets are short, typically under 140 characters
- The balanced dataset eliminates class imbalance concerns

---

## Data Preprocessing

The following preprocessing steps were applied:
- Conversion of text to lowercase
- Removal of URLs, user mentions, and retweet indicators
- Removal of numbers and punctuation
- Reduction of repeated characters
- Normalization of whitespace
- Optional stopword removal for classical ML models

The preprocessing ensures reduced noise while preserving sentiment-related information.

---

## Model Training

### Baseline Model
- **Text Representation:** TF-IDF (unigrams + bigrams)
- **Classifier:** Logistic Regression
- **Train/Test Split:** 80% / 20%

This baseline model provides strong performance for short and noisy text such as tweets.

---

## Results

- The model achieved strong accuracy for a baseline approach
- Precision and recall are balanced across both classes
- The confusion matrix shows no significant class bias

These results serve as a reliable baseline for comparison with more advanced models such as transformer-based architectures.

---

## Deployment

The trained model was deployed as an interactive **Streamlit web application** on Hugging Face Spaces.

### Features of the App:
- User-friendly text input
- Real-time sentiment prediction
- Display of cleaned input text
- Prediction confidence scores (if available)

> Note: On free-tier Hugging Face accounts, the Space may appear in a paused or sleeping state due to hardware quota limitations.

---

## Repository Structure

- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `Dockerfile` — Container configuration
- `logreg_tfidf_model.pkl` — Trained Logistic Regression model
- `tfidf_vectorizer.pkl` — Trained TF-IDF vectorizer
- `twitter-sentiment-analysis-using-sentiment140.ipynb` — Kaggle training notebook

---

## Future Work

- Fine-tuning transformer-based models such as BERT or DistilBERT
- Handling emojis and emoticons more effectively
- Multi-class sentiment classification (positive, neutral, negative)
- Expanding to multilingual sentiment analysis

---

## License

This project was developed for educational and academic purposes.
