import os
from datetime import datetime
import subprocess
import time

# Third-party imports
import numpy as np
import torch
from PIL import Image
import gradio as gr
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor
)

# Local imports
from qwen_vl_utils import process_vision_info

# Set to CPU only
device = "cpu"
print(f"[INFO] Using device: {device}")

# Disable CUDA to ensure CPU only operation
torch.cuda.is_available = lambda: False

# Initialize model and processor for CPU operation
try:
    print("[INFO] Loading Arabic-English-handwritten-OCR-v3 model on CPU...")
    
    # Load only one model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "sherif1313/Arabic-English-handwritten-OCR-v3", 
        trust_remote_code=True, 
        dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True
    ).eval()
    
    processor = AutoProcessor.from_pretrained(
        "sherif1313/Arabic-English-handwritten-OCR-v3", 
        trust_remote_code=True
    )
    
    print("[INFO] Model loaded successfully on CPU!")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    raise e

DESCRIPTION = """
# Arabic-English Text Extraction from Images (CPU Version)
This app uses the Arabic-English-handwritten-OCR-v3 model to extract Arabic and English text from images.
Runs on CPU only - it may be slower than the GPU version.
"""

def run_example(image):
    if image is None:
        return "⚠️ Please upload an image first", "0.00"
    
    try:
        start_time = time.time()
        
        print(f"[INFO] Processing image on CPU")

        # Handle image directly without saving
        if isinstance(image, np.ndarray):
            # If image is a numpy array, convert to PIL Image
            image_pil = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            # If image is already a PIL Image
            image_pil = image.convert("RGB")
        else:
            # If it's a file path
            image_pil = Image.open(image).convert("RGB")
        
        # Use a fixed prompt for Arabic/English text extraction
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_pil},
                    {"type": "text", "text": "ارجو استخراج النص العربي والإنجليزي كاملاً من هذه الصورة من البداية الى النهاية بدون اي اختصار ودون ذيادة او حذف. اقرأ كل المحتوى النصي الموجود في الصورة:"},
                ],
            }
        ]
        
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # Ensure inputs are on CPU
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        # Inference: Generation of the output
        with torch.no_grad():  # Save memory
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=1024,
                do_sample=False,  # Improve speed
                use_cache=True  # Improve speed
            )
        
        # Fix error: Use dictionary access instead of property access
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        end_time = time.time()
        total_time = round(end_time - start_time, 2)
        
        return output_text[0], f"{total_time} seconds"
    
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        return f"❌ An error occurred while processing the image: {str(e)}", "0.00"

# Create Gradio Interface
def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="Arabic-English Text Extraction from Images (CPU)") as demo:
        gr.Markdown(DESCRIPTION)
        
        with gr.Tab(label="Extract Arabic-English Text from Image"):
            with gr.Row():
                with gr.Column():
                    input_img = gr.Image(
                        label="Input Image",
                        type="pil",
                        height=300
                    )
                    submit_btn = gr.Button(
                        value="🔍 Extract Text",
                        variant="primary",
                        size="lg"
                    )
                    
                with gr.Column():
                    output_text = gr.Textbox(
                        label="Extracted Text (Running on CPU - might be slow)",
                        lines=10,
                        max_lines=15,
                        show_copy_button=True
                    )
                    time_taken = gr.Textbox(
                        label="Time Taken",
                        interactive=False
                    )
        
        # Examples for user
        gr.Examples(
            examples=[
                ["https://huggingface.co/sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2/resolve/main/assets/00002.png"],
                ["https://huggingface.co/sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2/resolve/main/assets/00106.png"],
                ["https://huggingface.co/sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2/resolve/main/assets/00107.png"],
                ["https://huggingface.co/sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2/resolve/main/assets/00113.png"],
                ["https://huggingface.co/sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2/resolve/main/assets/00126.png"],
                ["https://huggingface.co/sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2/resolve/main/assets/00135.png"],
                ["https://huggingface.co/sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2/resolve/main/assets/00141.png"],
                ["https://huggingface.co/sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2/resolve/main/assets/00197.png"],
                ["https://huggingface.co/sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2/resolve/main/assets/00198.png"],
                ["https://huggingface.co/sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2/resolve/main/assets/00199.png"],
                ["https://huggingface.co/sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2/resolve/main/assets/00216.png"],
                ["https://huggingface.co/sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2/resolve/main/assets/00240.png"],
            ],
            inputs=[input_img],
            label="Image Examples"
        )        
        # Bind events
        submit_btn.click(
            fn=run_example,
            inputs=[input_img],
            outputs=[output_text, time_taken]
        )
        
        # Add instructions
        with gr.Accordion("📋 Usage Instructions", open=False):
            gr.Markdown("""
            1. Upload an image containing Arabic or English text
            2. Click the 'Extract Text' button
            3. Wait for the extracted text to appear
            4. You can copy the resulting text using the copy button
            
            **Important notes for the CPU version:**
            - The app runs on the CPU only
            - Processing may be slow compared to the GPU version
            - It is recommended to use small images for better speed
            - Processing may take several minutes depending on image size
            
            **Model Information:**
            - Model: Arabic-English-handwritten-OCR-v3
            - This model supports both Arabic and English text extraction
            - The model is optimized for handwritten text recognition
            """)
    
    return demo

# Launch the app
if __name__ == "__main__":
    print("[INFO] Creating Gradio interface...")
    demo = create_interface()
    print("[INFO] Launching Gradio interface...")
    demo.launch(
        server_name="0.0.0.0" if os.getenv('SPACE_ID') else "127.0.0.1",
        share=False,
        debug=True,
        show_error=True
    )
