# 🎬 IMDb Movie Reviews Sentiment Analysis  

## 📌 Project Overview  
This project aims to classify IMDb movie reviews as **positive** or **negative** using **Deep Learning models**. It leverages **Natural Language Processing (NLP)** techniques such as **text preprocessing, tokenization, and word embeddings (GloVe)** to prepare the data for training **Shallow Neural Networks (SNN), Convolutional Neural Networks (CNN), and Long Short-Term Memory (LSTM) networks** for sentiment analysis.  

---

## 🚀 Features  
✅ **Cleans text data** (removes HTML tags, punctuation, stopwords)  
✅ **Tokenizes text** and converts it into numerical format  
✅ **Uses GloVe embeddings** for better word representation  
✅ **Implements multiple deep learning models:**  
   - Shallow Neural Network (**SNN**)  
   - Convolutional Neural Network (**CNN**)  
   - Long Short-Term Memory (**LSTM**)  
✅ **Evaluates model performance** with accuracy/loss curves  
✅ **Predicts sentiments** on unseen IMDb reviews  

---

## 🛠️ Installation  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

### 2️⃣ Install Dependencies  
Make sure you have Python 3.x installed, then install the required libraries:  

```bash
pip install numpy pandas tensorflow keras nltk matplotlib scikit-learn
```

### 3️⃣ Download GloVe Embeddings (100D)  
Download `glove.6B.100d.txt` from the GloVe website and place it in the project directory.  

---

## 📂 File Structure  
```bash
📂 imdb-sentiment-analysis  
 ├── 📜 a1_IMDB_Dataset.csv      # IMDb movie reviews dataset  
 ├── 📜 a2_glove.6B.100d.txt     # Pre-trained GloVe word embeddings  
 ├── 📜 a3_IMDb_Unseen_Reviews.csv # Sample unseen reviews for predictions  
 ├── 📜 sentiment_analysis.py    # Main script for training and evaluation  
 ├── 📜 README.md                # Project documentation  
 ├── 📜 requirements.txt         # Required dependencies  
```

---

## 🏗️ How It Works  

### 📌 Step 1: Load and Preprocess Data  
- Load IMDb reviews dataset (`a1_IMDB_Dataset.csv`)  
- Clean text (remove stopwords, special characters, and HTML tags)  
- Convert labels (positive → 1, negative → 0)  

### 📌 Step 2: Tokenization and Word Embeddings  
- Convert words into numerical representations using Keras Tokenizer  
- Load GloVe embeddings (100D) and create an embedding matrix  

### 📌 Step 3: Build & Train Models  
1️⃣ Shallow Neural Network (SNN)  
2️⃣ Convolutional Neural Network (CNN)  
3️⃣ Long Short-Term Memory (LSTM) Model  

- Each model is trained for **6 epochs** with **batch size = 128**  
- Models are evaluated on test data  

### 📌 Step 4: Evaluate Model Performance  
- Plot accuracy and loss curves  
- Compare SNN, CNN, and LSTM performance  

### 📌 Step 5: Make Predictions on Unseen Reviews  
- Load unseen reviews (`a3_IMDb_Unseen_Reviews.csv`)  
- Preprocess and tokenize new reviews  
- Predict sentiment using **LSTM model**  
- Save predictions to a CSV file  

---

## 🏆 Model Performance  
| Model | Test Accuracy |
|--------|--------------|
| SNN    | ~85%        |
| CNN    | ~88%        |
| LSTM   | ~90%        |  

🔹 **LSTM performed the best** in this project due to its ability to capture long-term dependencies in text.  

---

## 🔥 Results & Visualization  
📊 **Accuracy Curves:**  
📉 **Loss Curves:**  

---

## 💡 Future Improvements  
🚀 Improve accuracy with **Bidirectional LSTMs**  
🚀 Fine-tune **GloVe embeddings** using transfer learning  
🚀 Deploy model as a **Flask or FastAPI web app**  
🚀 Implement **real-time sentiment analysis** on user input  

---

## 📜 License  
This project is **open-source** and free to use. Contributions are welcome! 🚀  
