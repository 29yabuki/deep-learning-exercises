import gradio as gr
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_ID = "microsoft/Florence-2-large"
print(f"Loading model and processor: {MODEL_ID}...")

model_dtype = torch.float16 if device == 'cuda' else torch.float32
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=model_dtype
).to(device).eval()
print("Model and processor loaded.")

def extract_receipt_text_florence(image_input: Image.Image):
    task_prompt = '<OCR>'
    
    if image_input is None:
        return "Please upload an image."

    try:
        image = image_input.convert("RGB")

        inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device)
        inputs['pixel_values'] = inputs['pixel_values'].to(model_dtype)
        inputs['input_ids'] = inputs['input_ids'].to(device)


        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=4096,
                num_beams=3
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        image_size = (image_input.width, image_input.height)
        parsed_answer = processor.post_process_generation(
            generated_text, task=task_prompt, image_size=image_size
        )

        extracted_text = parsed_answer.get(task_prompt, "Error: Could not parse OCR output.")

        return extracted_text if extracted_text and isinstance(extracted_text, str) and extracted_text.strip() else "No text extracted or error processing."

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        if "CUDA out of memory" in str(e):
            return ("Error: GPU out of memory. Try a smaller image or restart runtime. "
                    "Ensure you're using float16 if possible.")
        return f"An error occurred during processing: {e}"

interface = gr.Interface(
    fn=extract_receipt_text_florence,
    inputs=gr.Image(type="pil", label="Upload receipt image"),
    outputs=gr.Textbox(label="Extracted text", lines=20),
    title="Receipt OCR with Florence-2",
    description=(
        "This application uses the microsoft/Florence-2-large model for text extraction."
    ),
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch(share=False)