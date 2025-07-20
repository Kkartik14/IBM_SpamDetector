# Spam Detection System

A machine learning-powered spam detection system built with Python and Gradio. This system uses a Naive Bayes classifier to identify spam messages with high accuracy.

## Features

- **Machine Learning Model**: Naive Bayes classifier with TF-IDF vectorization
- **Web Interface**: Clean and intuitive Gradio-based UI
- **High Accuracy**: Achieves 95%+ accuracy on SMS spam detection
- **Automatic Dataset**: Downloads and processes the SMS Spam Collection dataset
- **Real-time Predictions**: Instant spam/ham classification with confidence scores

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
   git clone https://github.com/Kkartik14/test-pr-reviewer.git
   cd test-pr-reviewer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Train the Model

Run the training script to download the dataset and train the spam detection model:

```bash
python train_model.py
```

This will:
- Download the SMS Spam Collection dataset from UCI repository
- Preprocess the text data
- Train a Naive Bayes classifier
- Save the trained model to `models/spam_model.pkl`

### Step 2: Launch the Web Interface

Start the Gradio web application:

```bash
python app.py
```

The web interface will be available at `http://localhost:7860`

## How It Works

### Machine Learning Pipeline

1. **Data Preprocessing**
   - Convert text to lowercase
   - Remove special characters and numbers
   - Normalize whitespace

2. **Feature Extraction**
   - TF-IDF vectorization with 3000 max features
   - English stop words removal

3. **Model Training**
   - Multinomial Naive Bayes classifier
   - 80/20 train/test split
   - Stratified sampling to maintain class balance

### Model Performance

- **Accuracy**: ~97%
- **Dataset**: 5,574 SMS messages
- **Classes**: Spam (13.4%) and Ham (86.6%)

## Example Usage

### Spam Examples:
- "Congratulations! You've won $1000! Click here to claim your prize now!"
- "URGENT: Your account will be suspended. Call this number immediately"

### Ham Examples:
- "Hey, are we still meeting for lunch tomorrow?"
- "The meeting has been moved to 3 PM in conference room B"

## Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms
- **gradio**: Web interface framework
- **numpy**: Numerical computing

### Model Architecture
- **Algorithm**: Multinomial Naive Bayes
- **Vectorizer**: TF-IDF with 3000 features
- **Preprocessing**: Text normalization and cleaning
- **Storage**: Pickle serialization

## Dataset

The system uses the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) from UCI Machine Learning Repository:

- **Source**: SMS Spam Collection v.1
- **Size**: 5,574 messages
- **Format**: Tab-separated values (label, message)
- **License**: Public domain

## Development Timeline

- **June 20, 2024**: Initial model training and dependencies
- **July 8, 2024**: Web interface implementation
- **Today**: Documentation and README

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UCI Machine Learning Repository for the SMS Spam Collection dataset
- Gradio team for the excellent web interface framework
- Scikit-learn community for machine learning tools