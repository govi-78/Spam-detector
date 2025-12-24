Spam Classification using Machine Learning


📌 Project Overview

Spam messages are a major concern across digital platforms such as SMS, emails, URLs, and social media. This project implements a Machine Learning–based spam detection system capable of identifying spam content from multiple sources with high accuracy.


🎯 Objectives

Detect spam messages across multiple platforms

Improve online safety and reduce fraudulent communication

Apply NLP and Machine Learning techniques for text classification


🧠 Technologies Used

Python

Machine Learning

Natural Language Processing (NLP)

Scikit-learn

Pandas, NumPy

TF-IDF / Bag of Words

Streamlit / Flask (optional UI)


📂 Project Structure
Spam-Classification/
│
├── data/
│   ├── sms_spam.csv
│   ├── email_spam.csv
│   └── url_spam.csv
│
├── model/
│   └── spam_classifier.pkl
│
├── notebooks/
│   └── training.ipynb
│
├── app.py
├── requirements.txt
└── README.md


⚙️ How the Project Works

Load datasets containing spam and non-spam messages

Preprocess text data (cleaning, tokenization, stop-word removal)

Convert text into numerical features using TF-IDF

Train machine learning models

Evaluate models and select the best one

Predict spam or non-spam for user input



▶️ How to Run the Project
Step 1: Clone the Repository
git clone https://github.com/govi-78/spam-classification.git
cd spam-classification

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Run the Application
python app.py




🧪 Sample Input
Congratulations! You have won a free lottery. Click here to claim now.

📤 Output
Spam



📊 Model Performance

Accuracy: ~95%

Precision: High

Recall: High

F1-score: Optimized for spam detection



🚀 Future Enhancements

Deep Learning models (LSTM, BERT)

Multilingual spam detection

Real-time spam filtering

Browser/email client integration



👨‍💻 Authors

Jnanashree TR
N Govind Prasad
Vibha Datta
