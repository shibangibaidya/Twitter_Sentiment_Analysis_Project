# Twitter Sentiment Analysis Using Machine Learning

This project focuses on building a **machine learning model** that analyzes and classifies tweets into one of three sentiment categories: **positive**, **negative**, or **neutral**. By applying **Natural Language Processing (NLP)** techniques, this project provides an effective tool for understanding the emotions behind user-generated content on Twitter.

---

## **Overview**

### Dataset
- **Total Tweets**: 162,980
- **Sentiment Classes**:
  - **Positive (1.0)**
  - **Neutral (0.0)**
  - **Negative (-1.0)**



---

## **Project Workflow**

### 1. **Data Preprocessing**
The dataset of tweets was prepared for analysis by implementing key preprocessing steps:
- **Cleaning the text**:
  - Removed special characters, punctuation, and emojis.
  - Converted text to lowercase for standardization.
- **Stopword Removal**:
  - Common stopwords were removed, except critical words like "not."
- **Tokenization**:
  - Tweets were tokenized into individual words for further analysis.

---

### 2. **Feature Extraction**
- Utilized **TF-IDF Vectorizer** to transform textual data into numerical feature vectors.
- Limited the feature space to the **5,000 most important words** (`max_features=5000`) to improve computational efficiency and focus on high-impact words.
- Explored unigrams (individual words) for initial analysis. Future steps might incorporate **bigrams** or **trigrams** for better context understanding.

---

### 3. **Model Training**
- Selected **Logistic Regression** as the classification algorithm due to its simplicity and robust performance for text classification tasks.
- Trained the model on the processed dataset using **balanced class weights** to handle any disparities in sentiment distribution.

---

### 4. **Evaluation Metrics**
#### **Training Data**:
- **Accuracy**: 90.39%
- **Class-wise Performance**:
  - Negative (-1.0): Precision = 0.83, Recall = 0.86, F1-score = 0.84
  - Neutral (0.0): Precision = 0.86, Recall = 0.97, F1-score = 0.91
  - Positive (1.0): Precision = 0.96, Recall = 0.85, F1-score = 0.90

#### **Test Data**:
- **Accuracy**: 89.27%
- **Class-wise Performance**:
  - Negative (-1.0): Precision = 0.80, Recall = 0.84, F1-score = 0.82
  - Neutral (0.0): Precision = 0.86, Recall = 0.97, F1-score = 0.91
  - Positive (1.0): Precision = 0.95, Recall = 0.83, F1-score = 0.89

---

### 5. **Prediction System**
- **User Input**:
  - The system accepts a tweet as input.
- **Feature Transformation**:
  - Converts input text into numerical features using the pre-trained TF-IDF Vectorizer.
- **Model Prediction**:
  - The logistic regression model predicts whether the tweet is **positive**, **negative**, or **neutral**.
- **Output**:
  - Displays the predicted sentiment category with a clear and concise message.

---

## **Visualization**
1. **Countplot**:
   - Visualizes the distribution of sentiments (positive, neutral, negative) across the dataset.
2. **Wordcloud**:
   - Generates a visual representation of the most frequently occurring words for each sentiment class, providing insights into dominant patterns in the text.

---

## **Tech Stack**

### **Programming Language**:
- Python

### **Libraries**:
- `numpy` and `pandas` for data manipulation
- `scikit-learn` for machine learning algorithms
- `nltk` (Natural Language Toolkit) for text preprocessing
- `matplotlib` and `seaborn` for visualizations
- `wordcloud` for generating word cloud visualizations
- `joblib` for saving and loading the trained model

---

## **Deployment**

To clone and run this project locally, follow these steps:

```bash
# Clone the repository
git clone https://github.com/shibangibaidya/Twitter_Sentiment_Analysis_Project.git

# Change to the project directory
cd Twitter-Sentiment-Analysis-Using-Machine-Learning

# Run the Jupyter Notebook
jupyter notebook twitter-sentiment-analysis.ipynb
