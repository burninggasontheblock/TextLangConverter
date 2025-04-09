# 🧠 Language Detection with Logistic Regression

This project is a simple yet powerful machine learning model that detects whether a piece of text is in **English** or **French** (or other languages, if added). It uses **TF-IDF character n-grams** to train a **Logistic Regression** classifier for high-accuracy language identification.

---

## 📦 What’s Inside?

- ✅ Text cleaning with regular expressions  
- ✅ TF-IDF Vectorization using character-level n-grams  
- ✅ Train/test split for evaluation  
- ✅ Logistic Regression for classification  
- ✅ Performance metrics: accuracy, classification report, confusion matrix

---

## 🛠️ How It Works

1. **Data Load:** Reads from `language.csv` (must have `text` and `language` columns).
2. **Preprocessing:** Lowercases the text and removes punctuation/numbers.
3. **Vectorization:** Uses `TfidfVectorizer` with character n-grams (2 to 4 chars).
4. **Model Training:** Logistic Regression classifier trained on 80% of the data.
5. **Evaluation:** Tests on the remaining 20% and prints detailed metrics.

---

## 🧪 Sample Output

```
Accuracy: 1.0
language
french     2000
english    2000

Classification Report:
              precision    recall  f1-score   support
     english       1.00      1.00      1.00       396
      french       1.00      1.00      1.00       404

Confusion Matrix:
[[396   0]
 [  0 404]]
```

---

## 📁 File Structure

```
language_detector/
│
├── main.py              # The main training & evaluation script
├── language.csv         # Your dataset (must include 'text' and 'language' columns)
└── README.md            # This file
```

---

## 🧠 Requirements

Install the necessary packages:

```bash
pip install pandas scikit-learn
```

---

## 📝 Data Format Example

Your `language.csv` file should look like this:

| text                         | language |
|-----------------------------|----------|
| Hello, how are you?         | english  |
| Bonjour, comment ça va ?    | french   |
| I love programming.         | english  |
| J'adore programmer.         | french   |

---

## 🚀 Run It

```bash
python main.py
```

---

## 🎯 Want to Improve It?

- Add more languages!
- Use a larger dataset
- Save/load the model with `joblib`
- Turn it into a web app with Flask or Streamlit

---

## 📬 Contact

Made with ❤️ for learning and experimenting.  
Feel free to reach out with feedback or suggestions!

---

Let me know if you want a version with emojis removed, or in more formal corporate tone instead!
