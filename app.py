import gradio as gr
import pickle
import re
import os

def load_model():
    with open("models/spam_model.pkl", "rb") as f:
        saved_data = pickle.load(f)
    return saved_data['vectorizer'], saved_data['model']

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_spam(message):
    if not message.strip():
        return "Please enter a message", "0.0000"
    
    if not os.path.exists("models/spam_model.pkl"):
        return "Model not found. Please train the model first by running: python train_model.py", "0.0000"
    
    try:
        vectorizer, model = load_model()
        processed_message = preprocess_text(message)
        message_vec = vectorizer.transform([processed_message])
        prediction = model.predict(message_vec)[0]
        probability = model.predict_proba(message_vec)[0]
        
        result = "SPAM" if prediction == 1 else "HAM"
        confidence = max(probability)
        
        return result, f"{confidence:.4f}"
    except Exception as e:
        return f"Error: {str(e)}", "0.0000"

with gr.Blocks(title="Spam Detection") as app:
    gr.Markdown("# Spam Detection System")
    gr.Markdown("Enter a message to check if it's spam or not.")
    
    with gr.Row():
        with gr.Column():
            message_input = gr.Textbox(
                lines=5,
                placeholder="Enter your message here...",
                label="Message"
            )
            predict_btn = gr.Button("Check Message", variant="primary")
        
        with gr.Column():
            result_output = gr.Textbox(label="Result", interactive=False)
            confidence_output = gr.Textbox(label="Confidence", interactive=False)
    
    predict_btn.click(
        fn=predict_spam,
        inputs=message_input,
        outputs=[result_output, confidence_output]
    )

if __name__ == "__main__":
    app.launch() 