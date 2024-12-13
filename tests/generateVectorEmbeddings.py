import os
import json
import logging
from typing import List
import openai

# Assuming ApiConfiguration is defined as shown, with the given constants:
from Apiconfiguration import ApiConfiguration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# Configuration parameters
CHUNK_SIZE = 1000    # Number of characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks
INPUT_DIR = "D:\Dissertation - City, Univeristy of London\Evaluating-AI-Learning-Assistants\input data\"  # Directory containing .txt files
OUTPUT_JSON = "data\embeddings_output.json"

# Initialize configuration
config = ApiConfiguration()

# Validate necessary environment and config
if not config.apiKey:
    logging.error("AZURE_OPENAI_API_KEY not set in environment.")
    raise EnvironmentError("Missing Azure OpenAI API key.")

if not config.azureEmbedDeploymentName:
    logging.error("Azure embedding deployment name not set in ApiConfiguration.")
    raise ValueError("Missing azureEmbedDeploymentName in ApiConfiguration.")

# Set the openai configuration for Azure
openai.api_type = "azure"

openai.api_base = "https://braidlms.openai.azure.com/"
openai.api_version = config.apiVersion
openai.api_key = config.apiKey

def get_embedding(text: str, config: ApiConfiguration):
    """
    Generate an embedding using Azure OpenAI embeddings.
    """
    text = text.replace("\n", " ")

    try:
        response = openai.Embedding.create(
            input=[text],
            engine=config.azureEmbedDeploymentName,  # Deployment name for embeddings
            timeout=config.openAiRequestTimeout
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        raise

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into chunks with the given size and overlap.
    """
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_length:
            break
        start += (chunk_size - overlap)
    return chunks

def process_directory(input_dir: str):
    """
    Process all .txt files in the specified directory and return a list of embeddings metadata.
    """
    embeddings_data = []

    if not os.path.isdir(input_dir):
        logging.error(f"Input directory '{input_dir}' does not exist.")
        return embeddings_data

    txt_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".txt")]
    if not txt_files:
        logging.warning(f"No .txt files found in directory '{input_dir}'.")
        return embeddings_data

    for filename in txt_files:
        file_path = os.path.join(input_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except Exception as e:
            logging.error(f"Failed to read file {file_path}: {e}")
            continue

        # Chunk the file
        logging.info(f"Chunking file: {filename}")
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        # Generate embeddings for each chunk
        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            try:
                embedding = get_embedding(chunk, config)
                embeddings_data.append({
                    "filename": filename,
                    "chunk_index": idx,
                    "chunk_text": chunk,
                    "embedding": embedding
                })
            except Exception as e:
                logging.error(f"Error embedding chunk {idx} of file {filename}: {e}")

    return embeddings_data

def main():
    logging.info("Starting embedding process...")
    embeddings_list = process_directory(INPUT_DIR)
    if not embeddings_list:
        logging.warning("No embeddings generated.")

    # Write embeddings to a JSON file
    try:
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as outfile:
            json.dump(embeddings_list, outfile, ensure_ascii=False, indent=4)
        logging.info(f"Embeddings successfully written to {OUTPUT_JSON}")
    except Exception as e:
        logging.error(f"Failed to write embeddings to JSON: {e}")

if __name__ == "__main__":
    main()
