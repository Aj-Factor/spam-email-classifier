# Spam Email Classifier

A beginner-friendly machine-learning project that classifies a text message as **spam** or **genuine (ham)**.

## Features
- TF-IDF text features and Logistic Regression classification.
- Streamlit web interface for testing messages.
- Training script with a held-out test set.
- Accuracy, precision, recall, F1-score and confusion matrix.
- Included CSV dataset, so no dataset download is needed to try the project.

## Dataset note
The included `data/sms_spam.csv` is a small educational sample dataset created for this project. It demonstrates the workflow but is not a production-grade benchmark. Do not present its evaluation score as real-world performance. For a stronger portfolio version, replace it with a larger, properly licensed SMS Spam Collection dataset and retrain.

## Requirements
- Python 3.10 or newer
- Internet connection for initial package installation

## Run in VS Code (Windows)
1. Extract the ZIP and open the `spam_email_classifier` folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Create a virtual environment:
   ```powershell
   py -m venv .venv
   ```
4. Activate it in PowerShell:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt and run:
   ```cmd
   .venv\Scripts\activate.bat
   ```
5. Install packages:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
6. Train and evaluate:
   ```bash
   python train_model.py
   ```
7. Start the web app:
   ```bash
   streamlit run app.py
   ```
Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Files
```text
spam_email_classifier/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── sms_spam.csv
```
The trained model (`spam_classifier.joblib`) and metrics report (`model_metrics.txt`) are generated when training runs. The model file is excluded from Git because it can be recreated.

## How it works
1. TF-IDF converts message text into numerical features.
2. Logistic Regression learns patterns from labelled examples.
3. The trained pipeline predicts `spam` or `ham`.
4. The training script reports test-set evaluation metrics.

## Limitations
- The included dataset is small and educational.
- Messages may be misclassified.
- Do not use this project as the only way to decide whether a message is safe.
- Avoid entering private messages into a publicly hosted demo.
