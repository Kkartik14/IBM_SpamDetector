import gradio as gr
import pickle
import re
import os

def load_email_model():
    with open("models/email_spam_model.pkl", "rb") as f:
        saved_data = pickle.load(f)
    return saved_data['vectorizer'], saved_data['model']

def preprocess_email_text(text):
    text = text.lower()
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_email_spam(email_content):
    if not email_content.strip():
        return "Please enter an email", "0.0000"
    
    if not os.path.exists("models/email_spam_model.pkl"):
        return "Model not found. Please train the model first by running: python train_model.py", "0.0000"
    
    try:
        vectorizer, model = load_email_model()
        processed_email = preprocess_email_text(email_content)
        email_vec = vectorizer.transform([processed_email])
        prediction = model.predict(email_vec)[0]
        probability = model.predict_proba(email_vec)[0]
        
        result = "SPAM" if prediction == 1 else "HAM"
        confidence = max(probability)
        
        return result, f"{confidence:.4f}"
    except Exception as e:
        return f"Error: {str(e)}", "0.0000"

with gr.Blocks(title="Email Spam Detection") as app:
    gr.Markdown("# Email Spam Detection System")
    gr.Markdown("Enter an email content (subject + body) to check if it's spam or legitimate (ham).")
    
    with gr.Row():
        with gr.Column():
            email_input = gr.Textbox(
                lines=8,
                placeholder="Enter email content here...\n\nExample:\nSubject: Congratulations! You've won!\n\nDear winner, you have won $1,000,000! Click here to claim your prize...",
                label="Email Content"
            )
            predict_btn = gr.Button("Check Email", variant="primary")
        
        with gr.Column():
            result_output = gr.Textbox(label="Result", interactive=False)
            confidence_output = gr.Textbox(label="Confidence", interactive=False)
    
    predict_btn.click(
        fn=predict_email_spam,
        inputs=email_input,
        outputs=[result_output, confidence_output]
    )
    
    gr.Markdown("### Example Emails to Test:")
    gr.Markdown("""
    **Spam Example:**
    ```
    Subject: URGENT: Claim your $1,000,000 prize!
    
    Congratulations! You have been selected as a winner in our international lottery!
    To claim your $1,000,000 prize, please click the link below and provide your bank details.
    This offer expires in 24 hours!
    ```
    
    **Ham Example:**
    ```
    Subject: Meeting rescheduled to Friday
    
    Hi team,
    
    I need to reschedule our weekly meeting from Thursday to Friday at 3 PM.
    Please let me know if this works for everyone.
    
    Best regards,
    John
    ```
    """)

if __name__ == "__main__":
    app.launch() 