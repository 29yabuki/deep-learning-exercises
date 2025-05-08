import gradio as gr
from google import genai
from dotenv import load_dotenv
import os
from PIL import Image

# load environment variables
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
model_name = "gemini-2.5-flash-preview-04-17"

prompt = "Answer the question DIRECTLY based on the receipt text only. Only provide an additional information if the question is about purchased items BUT MAKE IT REALLY SHORT AND SIMPLE. Avoid any unnecessary explanations or Markdown syntax. Provide a short and simple context to the question and answer. QUESTION:"

# load CSS
def load_css():
    try:
        with open('style.css', 'r') as file:
            return file.read()
    except FileNotFoundError:
        return ""

# OCR and QA
def answer_question(image: Image.Image, question: str):
    if image is None:
        return "ERROR: No image found."
    if not question:
        return "ERROR: No prompt."

    try:
        full_prompt = prompt + question
        response = client.models.generate_content(
            model = model_name,
            contents=[full_prompt, image]
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

# UI
with gr.Blocks(theme='NoCrypt/miku', css=load_css()) as demo:
    gr.Markdown("# ME 3")
    gr.Markdown("## QA on OCR")
    gr.Markdown("Please upload a receipt image and ask a question about the receipt.")
    with gr.Row():
        receipt_input = gr.Image(type="pil", label="Image input")
        question_input = gr.Textbox(label="Text input", placeholder="e.g., What is the total amount?")

    submit_btn = gr.Button("Process")
    answer_output = gr.Textbox(label="Answer", lines=3)

    submit_btn.click(fn=answer_question, inputs=[receipt_input, question_input], outputs=answer_output)

if __name__ == "__main__":
    demo.queue().launch(debug=True, share=False)