import gradio as gr
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer, pipeline
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import json
import os
from threading import Thread
import spaces
import pandas as pd

# Initialize NanoNets OCR2 3B model
model_name = "nanonets/Nanonets-OCR2-3B"
print(f"Loading OCR model: {model_name}")

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
).eval()

# Initialize orientation detection model
print("Loading orientation detection model...")
orientation_classifier = None
try:
    import timm
    from torchvision import transforms
    from huggingface_hub import hf_hub_download

    # Download model weights from HuggingFace
    # Based on: https://github.com/duartebarbosadev/deep-image-orientation-detection
    model_path = hf_hub_download(
        repo_id="DuarteBarbosa/deep-image-orientation-detection",
        filename="orientation_model_v2_0.9882.pth"
    )

    # Load the state dict
    checkpoint = torch.load(model_path, map_location='cpu')

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Print first few keys to debug
    sample_keys = list(state_dict.keys())[:5]
    print(f"Sample state_dict keys: {sample_keys}")

    # Remove common prefixes if present
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        # Strip common prefixes
        for prefix in ['model.', 'module.', '_orig_mod.']:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        new_state_dict[new_key] = value
    state_dict = new_state_dict

    # Try different EfficientNetV2 variants
    model_variants = [
        'tf_efficientnetv2_s.in21k',
        'tf_efficientnetv2_s',
        'efficientnetv2_rw_s',
        'tf_efficientnetv2_m',
        'efficientnet_b0',
        'efficientnet_b1',
    ]

    orientation_model = None
    for variant in model_variants:
        try:
            temp_model = timm.create_model(
                variant,
                pretrained=False,
                num_classes=4
            )
            missing, unexpected = temp_model.load_state_dict(state_dict, strict=False)
            if len(missing) < 50:  # Allow some missing keys
                orientation_model = temp_model
                print(f"✅ Successfully loaded with variant: {variant}")
                if missing:
                    print(f"  Missing keys: {len(missing)}")
                if unexpected:
                    print(f"  Unexpected keys: {len(unexpected)}")
                break
        except Exception as e:
            print(f"❌ Failed with {variant}: {str(e)[:100]}")
            continue

    if orientation_model is None:
        raise ValueError("Could not match state_dict to any EfficientNet variant")

    orientation_model.eval()

    if torch.cuda.is_available():
        orientation_model = orientation_model.to("cuda")

    # Define preprocessing transform (384x384 as per the original repo)
    orientation_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    orientation_classifier = {
        'model': orientation_model,
        'transform': orientation_transform
    }
    print("Orientation detection model (EfficientNetV2) loaded successfully")
except Exception as e:
    print(f"Warning: Could not load orientation model: {e}")
    print(f"Full error details: {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
    orientation_classifier = None

def correct_image_orientation(image):
    """
    Detect and correct image orientation using the orientation detection model.

    Args:
        image: PIL Image

    Returns:
        tuple: (corrected PIL Image, rotation info string)
    """
    if orientation_classifier is None:
        print("Orientation classifier not available, returning original image")
        return image, None

    try:
        model = orientation_classifier['model']
        transform = orientation_classifier['transform']

        # Prepare image for model
        img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        if torch.cuda.is_available():
            img_tensor = img_tensor.to("cuda")

        # Run inference
        with torch.no_grad():
            logits = model(img_tensor)
            predicted_class_idx = logits.argmax(-1).item()
            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
            confidence = probabilities[predicted_class_idx].item()

        # Map class to rotation angle
        # Class 0: 0°, Class 1: 90° clockwise, Class 2: 180°, Class 3: 270° clockwise (90° CCW)
        rotation_map = {
            0: 0,    # No rotation needed
            1: 270,  # 90° clockwise = 270° counter-clockwise
            2: 180,  # 180° rotation
            3: 90    # 270° clockwise = 90° counter-clockwise
        }

        rotation_angle = rotation_map.get(predicted_class_idx, 0)
        rotation_info = None

        if rotation_angle != 0:
            print(f"Rotating image by {rotation_angle}° (detected class: {predicted_class_idx}, confidence: {confidence:.1%})")
            image = image.rotate(rotation_angle, expand=True)
            rotation_info = f"🔄 **Image Auto-Corrected**: Rotated {rotation_angle}° counter-clockwise (confidence: {confidence:.1%})\n\n"
        else:
            print(f"Image orientation is correct, no rotation needed (confidence: {confidence:.1%})")
            rotation_info = f"✅ **Image Orientation**: Correct (confidence: {confidence:.1%})\n\n"

        return image, rotation_info

    except Exception as e:
        print(f"Error during orientation correction: {e}")
        import traceback
        traceback.print_exc()
        return image, None

@spaces.GPU
def extract_information(image, question, temperature=0.3, max_tokens=512, top_p=0.8, top_k=100, repetition_penalty=1.05, correct_orientation=True):
    """
    Extract information from an image based on a question with streaming support.

    Args:
        image: PIL Image or image path
        question: String question to ask about the image
        temperature: Sampling temperature (lower = more focused)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repeating tokens
        correct_orientation: Whether to auto-correct image orientation

    Yields:
        Streamed text tokens as they are generated
    """
    try:
        # Correct image orientation if enabled
        rotation_info = None
        if correct_orientation:
            image, rotation_info = correct_image_orientation(image)

        # Prepare the message with image and question
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ],
            }
        ]

        # Prepare inputs for the model
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
        inputs = inputs.to("cuda")

        # Setup streaming
        streamer = TextIteratorStreamer(
            processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
        )

        # Start generation in separate thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # Stream the output with rotation info prepended
        rotation_prefix = rotation_info if rotation_info else ""
        buffer = ""
        for new_text in streamer:
            if new_text and new_text != "<|im_end|>":
                buffer += new_text
                yield rotation_prefix + buffer

        thread.join()

    except Exception as e:
        yield f"Error: {str(e)}"

def gradio_interface(image, question, temperature, max_tokens, top_p, top_k, repetition_penalty, correct_orientation):
    """Gradio interface wrapper for streaming"""
    if image is None:
        yield "Please upload an image."
        return
    if not question.strip():
        yield "Please enter a question."
        return

    # Stream results from extract_information
    for partial_result in extract_information(image, question, temperature, max_tokens, top_p, top_k, repetition_penalty, correct_orientation):
        yield partial_result

def batch_process_images(files, questions_text, temperature, max_tokens, top_p, top_k, repetition_penalty, correct_orientation):
    """
    Process multiple images with multiple questions and return results in a table format

    Args:
        files: List of uploaded files
        questions_text: Text containing questions (one per line)
        temperature, max_tokens, etc: Generation parameters
        correct_orientation: Whether to auto-correct orientation

    Yields:
        Updated pandas DataFrame with results
    """
    if files is None or len(files) == 0:
        yield pd.DataFrame([["No files uploaded"]], columns=["Error"])
        return

    if not questions_text.strip():
        yield pd.DataFrame([["Please enter at least one question"]], columns=["Error"])
        return

    # Parse questions (one per line, ignore empty lines)
    questions = [q.strip() for q in questions_text.strip().split('\n') if q.strip()]

    if len(questions) == 0:
        yield pd.DataFrame([["Please enter at least one question"]], columns=["Error"])
        return

    # Create column headers: Image Name + one column per question
    columns = ["Image Name"] + [f"Q{i+1}: {q[:50]}..." if len(q) > 50 else f"Q{i+1}: {q}" for i, q in enumerate(questions)]

    # Initialize results list
    results = []

    for i, file in enumerate(files):
        try:
            # Open image
            image = Image.open(file.name)
            filename = os.path.basename(file.name)

            # Show processing status for all questions
            processing_row = [filename] + ["Processing..."] * len(questions)
            current_results = results + [processing_row]

            # Add queued entries for remaining files
            for remaining in files[i+1:]:
                queued_row = [os.path.basename(remaining.name)] + ["Queued"] * len(questions)
                current_results.append(queued_row)

            yield pd.DataFrame(current_results, columns=columns)

            # Process each question for this image
            answers = []
            for q_idx, question in enumerate(questions):
                # Extract information for this question
                final_result = ""
                for partial in extract_information(image, question, temperature, max_tokens, top_p, top_k, repetition_penalty, correct_orientation):
                    final_result = partial

                # Remove rotation info prefix if present to keep table clean
                if final_result.startswith("🔄") or final_result.startswith("✅"):
                    lines = final_result.split('\n')
                    if len(lines) > 2:
                        final_result = '\n'.join(lines[2:]).strip()

                answers.append(final_result)

                # Update table to show progress
                partial_row = [filename] + answers + ["Processing..."] * (len(questions) - len(answers))
                current_results = results + [partial_row]

                # Add queued entries for remaining files
                for remaining in files[i+1:]:
                    queued_row = [os.path.basename(remaining.name)] + ["Queued"] * len(questions)
                    current_results.append(queued_row)

                yield pd.DataFrame(current_results, columns=columns)

            # Add completed result for this image
            results.append([filename] + answers)

            # Update table with all completed results plus remaining queued
            remaining_results = results.copy()
            for remaining in files[i+1:]:
                queued_row = [os.path.basename(remaining.name)] + ["Queued"] * len(questions)
                remaining_results.append(queued_row)

            yield pd.DataFrame(remaining_results, columns=columns)

        except Exception as e:
            error_row = [filename if 'filename' in locals() else "Unknown"] + [f"Error: {str(e)}"] * len(questions)
            results.append(error_row)
            yield pd.DataFrame(results, columns=columns)

    # Final yield with all results
    yield pd.DataFrame(results, columns=columns)

# API endpoint function for MCP compatibility
def api_extract(image_path: str, question: str, temperature: float = 0.3, max_tokens: int = 512,
                top_p: float = 0.8, top_k: int = 100, repetition_penalty: float = 1.05,
                correct_orientation: bool = True):
    """
    API endpoint for extracting information from images.
    Can be called via HTTP POST.

    Args:
        image_path: Path to image or base64 encoded image
        question: Question to ask about the image
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        repetition_penalty: Repetition penalty
        correct_orientation: Whether to auto-correct image orientation

    Returns:
        JSON response with extracted information
    """
    try:
        if os.path.exists(image_path):
            image = Image.open(image_path)
        else:
            # Assume it's a base64 string
            import base64
            from io import BytesIO
            image_data = base64.b64decode(image_path)
            image = Image.open(BytesIO(image_data))

        # Consume the generator to get final result
        result = ""
        for partial in extract_information(image, question, temperature, max_tokens, top_p, top_k, repetition_penalty, correct_orientation):
            result = partial

        return {
            "success": True,
            "question": question,
            "answer": result,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "orientation_corrected": correct_orientation
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📄 OCR Information Extraction
        ### Powered by NanoNets OCR2 3B

        Extract specific information from documents and images using natural language questions.

        **Examples:**
        - "What is the name of the customer?"
        - "What is the total amount?"
        - "What is the invoice date?"
        - "Extract all the line items"
        """
    )

    with gr.Tabs():
        with gr.Tab("Single Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="pil", label="Upload Document/Image")
                    question_input = gr.Textbox(
                        label="Question",
                        placeholder="E.g., What is the name of the customer?",
                        lines=3
                    )

                    orientation_checkbox = gr.Checkbox(
                        label="Auto-correct image orientation",
                        value=True,
                        info="Automatically detect and rotate images to correct orientation before OCR"
                    )

                    with gr.Accordion("Advanced Options", open=False):
                        temperature_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            step=0.1,
                            label="Temperature (lower = more focused)"
                        )
                        max_tokens_slider = gr.Slider(
                            minimum=128,
                            maximum=2048,
                            value=512,
                            step=128,
                            label="Max Tokens"
                        )
                        top_p_slider = gr.Slider(
                            minimum=0.05,
                            maximum=1.0,
                            value=0.8,
                            step=0.05,
                            label="Top P (nucleus sampling)"
                        )
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=1000,
                            value=100,
                            step=10,
                            label="Top K"
                        )
                        repetition_penalty_slider = gr.Slider(
                            minimum=1.0,
                            maximum=2.0,
                            value=1.05,
                            step=0.05,
                            label="Repetition Penalty"
                        )

                    extract_btn = gr.Button("Extract Information", variant="primary")

                with gr.Column(scale=1):
                    output = gr.Textbox(label="Extracted Information", lines=10)

            # Example inputs
            gr.Examples(
                examples=[
                    ["What is the customer name?"],
                    ["What is the total amount?"],
                    ["What is the date on this document?"],
                    ["Extract the returner's name"],
                    ["What items are listed on this invoice?"],
                ],
                inputs=[question_input],
                label="Example Questions"
            )

            extract_btn.click(
                fn=gradio_interface,
                inputs=[image_input, question_input, temperature_slider, max_tokens_slider,
                        top_p_slider, top_k_slider, repetition_penalty_slider, orientation_checkbox],
                outputs=output
            )

        with gr.Tab("Batch Processing"):
            gr.Markdown(
                """
                ### 📦 Process Multiple Images with Multiple Questions

                Upload multiple images and ask multiple questions about each one.
                Each question creates a new column in the results table.

                **Enter one question per line:**
                - Line 1 = Column 1
                - Line 2 = Column 2
                - etc.
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    batch_files = gr.File(
                        file_count="multiple",
                        label="Upload Multiple Images",
                        file_types=["image"]
                    )
                    batch_question = gr.Textbox(
                        label="Questions (one per line)",
                        placeholder="What is the name of the customer?\nWhat is the total amount?\nWhat is the invoice date?",
                        lines=5
                    )

                    batch_orientation = gr.Checkbox(
                        label="Auto-correct image orientation",
                        value=True,
                        info="Automatically detect and rotate images to correct orientation before OCR"
                    )

                    with gr.Accordion("Advanced Options", open=False):
                        batch_temperature = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            step=0.1,
                            label="Temperature"
                        )
                        batch_max_tokens = gr.Slider(
                            minimum=128,
                            maximum=2048,
                            value=512,
                            step=128,
                            label="Max Tokens"
                        )
                        batch_top_p = gr.Slider(
                            minimum=0.05,
                            maximum=1.0,
                            value=0.8,
                            step=0.05,
                            label="Top P"
                        )
                        batch_top_k = gr.Slider(
                            minimum=1,
                            maximum=1000,
                            value=100,
                            step=10,
                            label="Top K"
                        )
                        batch_repetition_penalty = gr.Slider(
                            minimum=1.0,
                            maximum=2.0,
                            value=1.05,
                            step=0.05,
                            label="Repetition Penalty"
                        )

                    batch_btn = gr.Button("Process All Images", variant="primary")

                with gr.Column(scale=1):
                    batch_output = gr.Dataframe(
                        label="Results",
                        wrap=True,
                        interactive=False
                    )

            batch_btn.click(
                fn=batch_process_images,
                inputs=[batch_files, batch_question, batch_temperature, batch_max_tokens,
                        batch_top_p, batch_top_k, batch_repetition_penalty, batch_orientation],
                outputs=batch_output
            )

# Launch with MCP server enabled for full MCP compatibility
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=True,      # Enable API documentation
        mcp_server=True     # Enable MCP server for AI assistants
    )
