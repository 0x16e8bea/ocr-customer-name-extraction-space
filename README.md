---
title: OCR Information Extraction
emoji: 📄
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
---

# OCR Information Extraction

Extract specific information from documents and images using natural language questions powered by NanoNets OCR2 3B model.

## Features

- 🎯 **Question-based Extraction**: Ask natural language questions to extract specific information
- 📄 **Document Understanding**: Works with invoices, receipts, forms, and various documents
- 📦 **Batch Processing**: Upload and process multiple images at once with results in a table
- 🔄 **Auto Orientation Correction**: Automatically detects and corrects image rotation (0°, 90°, 180°, 270°)
- 🔌 **API Access**: RESTful API endpoint for programmatic access
- 🌐 **MCP Compatible**: Can be integrated with Model Context Protocol applications
- ⚡ **GPU Accelerated**: Uses ZeroGPU for fast inference
- 📡 **Streaming Responses**: See results in real-time as they generate

## Usage

### Web Interface

#### Single Image Mode

1. Upload a document or image
2. Enter your question (e.g., "What is the name of the customer?")
3. (Optional) Toggle "Auto-correct image orientation" on/off
4. Click "Extract Information"

#### Batch Processing Mode

Process multiple images with multiple questions at once:

1. Switch to the "Batch Processing" tab
2. Upload multiple images (drag & drop or select multiple files)
3. Enter your questions - **one question per line**:
   ```
   What is the name of the customer?
   What is the total amount?
   What is the invoice date?
   ```
4. Click "Process All Images"
5. Results appear in a table with:
   - **Image Name**: Filename of each image
   - **Q1, Q2, Q3...**: One column per question with answers
   - **Status Updates**: See "Queued" → "Processing..." → Final result

**Example Output:**

| Image Name | Q1: What is the customer name? | Q2: What is the total amount? | Q3: What is the date? |
|------------|-------------------------------|------------------------------|---------------------|
| invoice1.jpg | John Doe | $150.00 | 2025-01-15 |
| invoice2.jpg | Jane Smith | $275.50 | 2025-01-16 |
| invoice3.jpg | Bob Wilson | $98.75 | 2025-01-17 |

Perfect for:
- Processing multiple invoices to extract multiple fields at once
- Batch extraction from receipts (merchant, amount, date)
- Analyzing forms for all required information
- Creating structured data from unstructured documents

#### Image Orientation Correction

The space includes automatic image orientation detection and correction using the [DuarteBarbosa/deep-image-orientation-detection](https://huggingface.co/DuarteBarbosa/deep-image-orientation-detection) model (98.82% accuracy). This is especially useful for:
- Scanned documents that were rotated
- Photos taken at incorrect angles
- Documents with mixed orientations

The orientation correction is **enabled by default** but can be toggled off if not needed.

**Visual Feedback**: When an image is rotated, you'll see a message like:
- 🔄 **Image Auto-Corrected**: Rotated 90° counter-clockwise (confidence: 95.2%)
- ✅ **Image Orientation**: Correct (no rotation needed)

This helps you understand if and how the image was adjusted before OCR processing.

### API Endpoint

The space exposes a RESTful API that can be accessed programmatically:

#### Using Python

```python
from gradio_client import Client

client = Client("YOUR_SPACE_URL")
result = client.predict(
    image="/path/to/image.jpg",
    question="What is the name of the customer?",
    temperature=0.3,
    max_tokens=512,
    api_name="/predict"
)
print(result)
```

#### Using cURL

```bash
curl -X POST "YOUR_SPACE_URL/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"path": "/path/to/image.jpg"},
      "What is the customer name?",
      0.3,
      512
    ]
  }'
```

## MCP Integration

This space is **MCP (Model Context Protocol) compatible** and can be used directly with AI assistants like Claude Desktop, Cursor, VS Code, and Zed.

### MCP Endpoint

Once deployed, the MCP server is automatically available at:
```
https://0x16e8bea-ocr-customer-name-extraction-space.hf.space/gradio_api/mcp/sse
```

### Connect to Claude Desktop

1. Go to your Claude Desktop MCP configuration:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

2. Add this configuration:

```json
{
  "mcpServers": {
    "ocr-extraction": {
      "transport": "sse",
      "url": "https://0x16e8bea-ocr-customer-name-extraction-space.hf.space/gradio_api/mcp/sse"
    }
  }
}
```

3. Restart Claude Desktop

### Using with Claude

Once connected, you can ask Claude:
> "Use the OCR tool to extract the customer name from this invoice"

Claude will automatically use the `gradio_interface` tool exposed by this Space.

## Example Questions

- "What is the name of the customer?"
- "What is the total amount?"
- "What is the invoice date?"
- "Extract the returner's name"
- "What items are listed on this invoice?"
- "What is the company name?"
- "What is the address?"

## Model

This space uses the [NanoNets OCR2 3B](https://huggingface.co/nanonets/Nanonets-OCR2-3B) model, which is based on Qwen2.5-VL and optimized for OCR and document understanding tasks with high accuracy.

## Parameters

- **Temperature** (0.0-1.0): Controls randomness. Lower values make output more focused and deterministic.
- **Max Tokens** (128-2048): Maximum length of the generated response.

## Deployment

### Local Deployment

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/ocr-extraction-space
cd ocr-extraction-space
pip install -r requirements.txt
python app.py
```

### Hugging Face Spaces

This repository is ready to be deployed on Hugging Face Spaces. Simply create a new Space and push this repository.

## License

Apache 2.0