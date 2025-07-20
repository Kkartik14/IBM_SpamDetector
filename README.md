# Email Spam Detection System

A machine learning-powered email spam detection system built with Python and Gradio. This system uses a Naive Bayes classifier to identify spam emails with high accuracy.

## Features

- **Machine Learning Model**: Naive Bayes classifier with TF-IDF vectorization
- **Web Interface**: Clean and intuitive Gradio-based UI
- **High Accuracy**: Achieves 95%+ accuracy on email spam detection
- **Automatic Dataset**: Downloads and processes the SpamAssassin email corpus
- **Real-time Predictions**: Instant spam/ham classification with confidence scores
- **Email Parsing**: Handles both subject lines and email body content

## Project Structure

```
test-pr-reviewer/
├── requirements.txt    # Python dependencies
├── train_model.py     # ML model training script
├── app.py            # Gradio web interface
├── .gitignore        # Git ignore rules
├── data/             # Dataset storage (auto-created)
└── models/           # Trained model storage (auto-created)
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kkartik14/IBM_SpamDetector.git
   cd IBM_SpamDetector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Train the Model

Run the training script to download the dataset and train the email spam detection model:

```bash
python train_model.py
```

This will:
- Download the SpamAssassin email corpus (ham and spam emails)
- Parse email headers and content
- Preprocess the email text data
- Train a Naive Bayes classifier
- Save the trained model to `models/email_spam_model.pkl`

### Step 2: Launch the Web Interface

Start the Gradio web application:

```bash
python app.py
```

The web interface will be available at `http://localhost:7860`

## How It Works

### Machine Learning Pipeline

1. **Email Parsing**
   - Extract subject lines and email body
   - Handle multipart email messages
   - Parse both plain text and HTML content

2. **Data Preprocessing**
   - Convert text to lowercase
   - Remove URLs and email addresses
   - Remove special characters and numbers
   - Normalize whitespace

3. **Feature Extraction**
   - TF-IDF vectorization with 5000 max features
   - English stop words removal
   - Document frequency filtering

4. **Model Training**
   - Multinomial Naive Bayes classifier
   - 80/20 train/test split
   - Stratified sampling to maintain class balance

### Model Performance

- **Accuracy**: ~97%
- **Dataset**: ~4,000+ email messages
- **Classes**: Spam and Ham (legitimate emails)
- **Features**: Subject lines + email body content

## Example Usage

### Spam Email Example:
```
Subject: URGENT: Claim your $1,000,000 prize!

Congratulations! You have been selected as a winner in our international lottery!
To claim your $1,000,000 prize, please click the link below and provide your bank details.
This offer expires in 24 hours!
```

### Ham Email Example:
```
Subject: Meeting rescheduled to Friday

Hi team,

I need to reschedule our weekly meeting from Thursday to Friday at 3 PM.
Please let me know if this works for everyone.

Best regards,
John
```

## Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms
- **gradio**: Web interface framework
- **numpy**: Numerical computing

### Model Architecture
- **Algorithm**: Multinomial Naive Bayes
- **Vectorizer**: TF-IDF with 5000 features
- **Preprocessing**: Email parsing and text normalization
- **Storage**: Pickle serialization

## Dataset

The system uses the [SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/):

- **Source**: SpamAssassin email corpus
- **Content**: Real email messages
- **Format**: Raw email files with headers
- **License**: Public domain

## Development Timeline

- **June 20, 2025**: Initial model training and dependencies
- **June 28, 2025**: Model training completed for spam detection with >97% accuracy
- **July 8, 2025**: Web interface implementation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- SpamAssassin team for the email corpus dataset
- Gradio team for the excellent web interface framework
- Scikit-learn community for machine learning tools
