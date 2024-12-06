# Copyright (c) 2024 Braid Technologies Ltd

"""
Question Generation (Persona):
Keeps using `gpt-4o` for persona-based question generation.
    
Similarity Embedding:
Upgraded to `text-embedding-3-large` for improved embedding precision and similarity calculations.
    
Evaluation LLM:
Continues with `gemini-1.5-pro` for evaluating the responses generated by GPT-4o.
"""

# Standard Library Imports
import logging
import os
import json
import sys
from logging import Logger
from typing import List, Dict, Any
import numpy as np
from numpy.linalg import norm
import datetime
import csv  # Add this import at the top of the file


# Third-Party Packages
from openai import AzureOpenAI, OpenAIError, BadRequestError, APIConnectionError
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type
from GeminiEvaluator import GeminiEvaluator
from GPT4oEvaluator import GPT4oEvaluator


# Add the project root and scripts directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Local Modules
from common.ApiConfiguration import ApiConfiguration
from common.common_functions import get_embedding
from common.common_functions import call_openai_chat
from PersonaStrategy import DeveloperPersonaStrategy, TesterPersonaStrategy, BusinessAnalystPersonaStrategy, PersonaStrategy

# Constants
SIMILARITY_THRESHOLD = 0.8          # Defines the minimum similarity threshold for a question to be considered a hit
MAX_RETRIES = 15                    # Maximum number of retries for API calls
NUM_QUESTIONS = 100                 # Number of questions to be generated per test

# OpenAI prompts used for persona generation, enrichment, and follow-up question generation
OPENAI_PERSONA_PROMPT =  "You are an AI assistant helping an application developer understand generative AI. You explain complex concepts in simple language, using Python examples if it helps. You limit replies to 50 words or less. If you don't know the answer, say 'I don't know'. If the question is not related to building AI applications, Python, or Large Language Models (LLMs), say 'That doesn't seem to be about AI'."
ENRICHMENT_PROMPT = "You will be provided with a question about building applications that use generative AI technology. Write a 50 word summary of an article that would be a great answer to the question. Consider enriching the question with additional topics that the question asker might want to understand. Write the summary in the present tense, as though the article exists. If the question is not related to building AI applications, Python, or Large Language Models (LLMs), say 'That doesn't seem to be about AI'.\n"
FOLLOW_UP_PROMPT =  "You will be provided with a summary of an article about building applications that use generative AI technology. Write a question of no more than 10 words that a reader might ask as a follow up to reading the article."
FOLLOW_UP_ON_TOPIC_PROMPT = "You are an AI assistant helping a team of developers understand AI. You explain complex concepts in simple language. Respond 'yes' if the follow-up question is about AI, otherwise respond 'no'."

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

gemini_evaluator = GeminiEvaluator()  ## Initialize an instance of the GeminiEvaluator for response evaluation

# Function to configure the Azure OpenAI API client
def configure_openai_for_azure(config: ApiConfiguration, task: str) -> AzureOpenAI:
    """
    Configures OpenAI for Azure using the provided ApiConfiguration.
    
    Args:
        config (ApiConfiguration): The ApiConfiguration object containing the necessary settings.
        task (str): The task for which OpenAI is being configured ("chat" or "embedding").
        
    Returns:
        AzureOpenAI: An instance of AzureOpenAI configured with the correct settings.
    """
    if task == "chat":
        return AzureOpenAI(
            azure_endpoint=config.resourceChatCompletionEndpoint,
            api_key=config.apiKey.strip(),
            api_version=config.apiVersion
        )
    elif task == "embedding":
        return AzureOpenAI(
            azure_endpoint=config.resourceEmbeddingEndpoint,
            api_key=config.apiKey.strip(),
            api_version=config.apiVersion
        )

# Class to hold test results
class TestResult:
    def __init__(self) -> None:

        """
        Initializes a new instance of the TestResult class.

        Sets the initial state of the test result, including the question, enriched question, 
        hit status, hit relevance, hit summary, follow-up question, and follow-up topic.
        
        Args:
            None
        
        Returns:
            None
        """
        self.question: str = ""                             # Original question
        self.hit_relevance: float = 0.0                     # Relevance score of the hit
        self.enriched_question_summary: str = ""            # Summary of the enriched question   
        self.hit: bool = False                              # Whether the question was considered a hit based on similarity
        self.hit_relevance: float = 0.0                     # Relevance score of the hit
        self.follow_up: str = ""                            # Adding followUp field
        self.follow_up_on_topic: str = ""                   # Adding followUpOnTopic field
        self.gemini_evaluation: str = ""                    # Field to store Gemini LLM evaluation
        self.gpt4o_evaluation: str = ""                     # Field to store GPT-4o evaluation

# # Function to call the OpenAI API with retry logic
# @retry(wait=wait_random_exponential(min=5, max=15), stop=stop_after_attempt(MAX_RETRIES), retry=retry_if_not_exception_type(BadRequestError))
# def call_openai_chat(chat_client: AzureOpenAI, messages: List[Dict[str, str]], config: ApiConfiguration, logger: logging.Logger) -> str:
#     """
#     Retries the OpenAI chat API call with exponential backoff and retry logic.

#     :param chat_client: An instance of the AzureOpenAI class.
#     :type chat_client: AzureOpenAI
#     :param messages: A list of dictionaries representing the messages to be sent to the API.
#     :type messages: List[Dict[str, str]]
#     :param config: An instance of the ApiConfiguration class.
#     :type config: ApiConfiguration
#     :param logger: An instance of the logging.Logger class.
#     :type logger: logging.Logger
#     :return: The content of the first choice in the API response.
#     :rtype: str
#     :raises RuntimeError: If the finish reason in the API response is not 'stop', 'length', or an empty string.
#     :raises OpenAIError: If there is an error with the OpenAI API.
#     :raises APIConnectionError: If there is an error with the API connection.
#     """
#     try:
#         response = chat_client.chat.completions.create(
#             model=config.azureDeploymentName,
#             messages=messages,
#             temperature=0.7,
#             max_tokens=config.maxTokens,
#             top_p=0.0,
#             frequency_penalty=0,
#             presence_penalty=0,
#             timeout=config.openAiRequestTimeout,
#         )
#         content = response.choices[0].message.content
#         finish_reason = response.choices[0].finish_reason

#         if finish_reason not in {"stop", "length", ""}:
#             logger.warning("Unexpected stop reason: %s", finish_reason)
#             logger.warning("Content: %s", content)
#             logger.warning("Consider increasing max tokens and retrying.")
#             raise RuntimeError("Unexpected finish reason in API response.")

#         return content

#     except (OpenAIError, APIConnectionError) as e:
#         logger.error(f"Error: {e}")
#         raise


# Function to retrieve text embeddings using OpenAI API with retry logic
@retry(wait=wait_random_exponential(min=5, max=15), stop=stop_after_attempt(MAX_RETRIES), retry=retry_if_not_exception_type(BadRequestError))
def get_text_embedding(embedding_client: AzureOpenAI, config: ApiConfiguration, text: str, logger: Logger) -> np.ndarray:
    """
    Retrieves the text embedding for a given text using the OpenAI API.

    Args:
        embedding_client (AzureOpenAI): The OpenAI client instance.
        config (ApiConfiguration): The API configuration instance.
        text (str): The text for which to retrieve the embedding.
        logger (Logger): The logger instance.

    Returns:
        np.ndarray: The text embedding as a numpy array.

    Raises:
        OpenAIError: If an error occurs while retrieving the text embedding.
    """
    try:
        embedding = get_embedding(text, embedding_client, config)
        return np.array(embedding)
    except OpenAIError as e:
        logger.error(f"Error getting text embedding: {e}")
        raise

# Function to calculate cosine similarity between two vectors
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two vectors.

    Args:
        a (np.ndarray): The first vector.
        b (np.ndarray): The second vector.

    Returns:
        float: The cosine similarity between the two vectors.

    Raises:
        ValueError: If the input vectors are not numpy arrays or convertible to numpy arrays.
        ValueError: If the input vectors do not have the same shape.
        ValueError: If either of the input vectors is a zero vector.
    """
    try:
        a, b = np.array(a), np.array(b)
    except Exception:
        raise ValueError("Input vectors must be numpy arrays or convertible to numpy arrays")

    if a.shape != b.shape:
        raise ValueError("Input vectors must have the same shape")

    dot_product = np.dot(a, b)
    a_norm, b_norm = norm(a), norm(b)

    if a_norm == 0 or b_norm == 0:
        raise ValueError("Input vectors must not be zero vectors")

    return dot_product / (a_norm * b_norm)

# Function to generate enriched questions using OpenAI API
@retry(wait=wait_random_exponential(min=5, max=15), stop=stop_after_attempt(MAX_RETRIES), retry=retry_if_not_exception_type(BadRequestError))
def generate_enriched_question(chat_client: AzureOpenAI, config: ApiConfiguration, question: str, logger: logging.Logger) -> str:
    """
    Generates an enriched question using the OpenAI API.

    Args:
        chat_client (AzureOpenAI): The OpenAI client instance.
        config (ApiConfiguration): The API configuration instance.
        question (str): The question to be enriched.
        logger (logging.Logger): The logger instance.

    Returns:
        str: The enriched question.

    Raises:
        BadRequestError: If the API request fails.
    """
    messages = [
        {"role": "system", "content": OPENAI_PERSONA_PROMPT},
        {"role": "user", "content": ENRICHMENT_PROMPT + "Question: " + question},
    ]
    logger.info("Making API request to OpenAI...")
    logger.info("Request payload: %s", messages)

    response = call_openai_chat(chat_client, messages, config, logger)
    logger.info("API response received: %s", response)

    return response


def generate_follow_up_question(chat_client: AzureOpenAI, config: ApiConfiguration, text: str, logger: logging.Logger) -> str:
    """
    Generates a follow-up question using the OpenAI API.

    Args:
        chat_client (AzureOpenAI): The OpenAI client instance.
        config (ApiConfiguration): The API configuration instance.
        text (str): The text to generate a follow-up question about.
        logger (logging.Logger): The logger instance.

    Returns:
        str: The follow-up question.

    Raises:
        BadRequestError: If the API request fails.
    """
    messages = [
        {"role": "system", "content": FOLLOW_UP_PROMPT},
        {"role": "user", "content": text},
    ]
    response = call_openai_chat(chat_client, messages, config, logger)
    return response


def assess_follow_up_on_topic(chat_client: AzureOpenAI, config: ApiConfiguration, follow_up: str, logger: logging.Logger) -> str:
    """
    Checks if a follow-up question is about AI using the OpenAI API.

    Args:
        chat_client (AzureOpenAI): The OpenAI client instance.
        config (ApiConfiguration): The API configuration instance.
        follow_up (str): The follow-up question to assess.
        logger (logging.Logger): The logger instance.

    Returns:
        str: 'yes' if the follow-up question is about AI, 'no' otherwise.

    Raises:
        BadRequestError: If the API request fails.
    """
    messages = [
        {"role": "system", "content": FOLLOW_UP_ON_TOPIC_PROMPT},
        {"role": "user", "content": follow_up},
    ]
    response = call_openai_chat(chat_client, messages, config, logger)
    return response

def process_questions(chat_client: AzureOpenAI, embedding_client: AzureOpenAI, config: ApiConfiguration, questions: List[str], processed_question_chunks: List[Dict[str, Any]], logger: logging.Logger, llm_choice: str = '1') -> List[TestResult]:
    question_results: List[TestResult] = []
    
    for question in questions:
        question_result = TestResult()
        question_result.question = question

        question_result.enriched_question_summary = generate_enriched_question(chat_client, config, question, logger)
        
        embedding = get_text_embedding(embedding_client, config, question_result.enriched_question_summary, logger)

        best_hit_relevance = 0
        best_hit_summary = None

        for chunk in processed_question_chunks:
            if chunk and isinstance(chunk, dict):
                gpt4_embedding = chunk.get("embedding")
                similarity = cosine_similarity(gpt4_embedding, embedding)

                if similarity > SIMILARITY_THRESHOLD:
                    question_result.hit = True

                if similarity > best_hit_relevance:
                    best_hit_relevance = similarity
                    best_hit_summary = chunk.get("summary")
        
        question_result.hit_relevance = best_hit_relevance
        question_result.hit_summary = best_hit_summary

        if question_result.hit_summary:
            question_result.follow_up = generate_follow_up_question(chat_client, config, question_result.hit_summary, logger)
            question_result.follow_up_on_topic = assess_follow_up_on_topic(chat_client, config, question_result.follow_up, logger)
        
        if llm_choice == '1':
            question_result.gpt4o_evaluator = GPT4oEvaluator.evaluate(
                question_result.question,
                question_result.enriched_question_summary
            )
        elif llm_choice == '2':
            question_result.gemini_evaluation = gemini_evaluator.evaluate(
                question_result.question,
                question_result.enriched_question_summary
            )

        question_results.append(question_result)

    logger.debug("Total tests processed: %s", len(question_results))

    return question_results

# Function to read processed chunks from the source directory
def read_processed_chunks(source_dir: str) -> List[Dict[str, Any]]:
    """
    Reads and processes JSON files from a specified source directory.

    Args:
        source_dir (str): The path to the source directory containing JSON files.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the processed JSON data.

    Raises:
        FileNotFoundError: If the source directory or a JSON file is not found.
        IOError: If an I/O error occurs while reading a JSON file.
    """
    processed_question_chunks: List[Dict[str, Any]] = []            # Initialize an empty list to hold the chunks.
    try:
        # Loop through all files in the specified directory.
        for filename in os.listdir(source_dir):
            # Check if the file has a '.json' extension.
            if filename.endswith(".json"):
                file_path = os.path.join(source_dir, filename)      # Get the full path to the file.
                # Open the file and load its contents as JSON.
                with open(file_path, "r", encoding="utf-8") as f:
                    chunk = json.load(f)                        # Load JSON data into the chunk variable.
                    processed_question_chunks = chunk           # Store the JSON content in the processed chunks list.

    # Handle file not found or I/O errors that occur during file reading.                
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Error reading files: {e}")
        raise
    
    # If no chunks were processed, log a warning.
    if not processed_question_chunks:
        logger.error("Processed question chunks are None or empty.")
    
    return processed_question_chunks

# Function to save the results and generated questions
def save_results(test_destination_dir: str, question_results: List[TestResult], test_mode: str) -> None:
    """
    Saves the test results to a JSON file in the specified destination directory.

    Args:
        test_destination_dir (str): The path to the directory where the test results will be saved.
        question_results (List[TestResult]): A list of TestResult objects containing the test results.
        test_mode (str): The test mode to be used in the output file name.

    Returns:
        None

    Raises:
        IOError: If an I/O error occurs while writing the JSON file.
    """
    output_data = [
        {
            "question": result.question,                                # Original question.
            "enriched_question": result.enriched_question_summary,      # Enriched question summary.
            "hit": result.hit,                                          # Whether it was a hit or not (based on similarity).
            "summary": result.hit_summary,                              # The best-matching pre-processed summary.
            "hitRelevance": result.hit_relevance,                       # Relevance score for the best hit.
            "follow_up": result.follow_up,                              # Follow-up question generated.
            "follow_up_on_topic": result.follow_up_on_topic,            # Whether the follow-up is on-topic.
            "gemini_evaluation": result.gemini_evaluation,               # Evaluation result from Gemini.
            "gpt4o_evaluation": result.gpt4o_evaluation                # Evaluation result from GPT-4o
        }
        for result in question_results                                  # Iterate over each TestResult and serialize it.
    ]

    # Generate a unique filename for the output based on the current timestamp and test mode.
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(test_destination_dir, f"test_output_v5_{test_mode}_{current_datetime}.csv")

    try:
        with open(output_file, "w", encoding="utf-8", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "question", "enriched_question", "hit", "summary",
                "hitRelevance", "follow_up", "follow_up_on_topic", "gemini_evaluation"
            ])
            writer.writeheader()
            for result in question_results:
                writer.writerow({
                    "question": result.question,
                    "enriched_question": result.enriched_question_summary,
                    "hit": result.hit,
                    "summary": result.hit_summary,
                    "hitRelevance": result.hit_relevance,
                    "follow_up": result.follow_up,
                    "follow_up_on_topic": result.follow_up_on_topic,
                    "gemini_evaluation": result.gemini_evaluation
                })
        logger.info(f"Test results saved to: {output_file}")
    except IOError as e:
        logger.error(f"Error saving results: {e}")
        raise

# Main test-running function
def run_tests(config: ApiConfiguration, test_destination_dir: str, source_dir: str, num_questions: int = 100, questions: List[str] = None, persona_strategy: PersonaStrategy = None, llm_choice: str = '1') -> None:
    """
    Runs tests using the provided configuration, test destination directory, source directory, and questions.

    Args:
        config (ApiConfiguration): The configuration for the API.
        test_destination_dir (str): The path to the directory where the test results will be saved.
        source_dir (str): The directory containing the source files.
        num_questions (int): The number of questions to generate using the persona strategy.
        questions (List[str]): A list of questions to be processed.
        persona_strategy (PersonaStrategy): The persona strategy to use for generating questions.
        llm_choice (str): The choice of LLM for persona generation and evaluation ('1' for GPT-4o, '2' for Gemini-1.5-pro).

    Returns:
        None
    """
    # Initialize the OpenAI clients for both chat completions and embeddings.
    chat_client = configure_openai_for_azure(config, "chat")
    embedding_client = configure_openai_for_azure(config, "embedding")

    # Ensure that a test destination directory is provided, raise an error if not.
    if not test_destination_dir:
        logger.error("Test data folder not provided")                       # Log error message.
        raise ValueError("Test destination directory not provided")         # Raise exception

    if persona_strategy:
        if llm_choice == '1':
            # Use GPT-4o for persona generation
            questions = persona_strategy.generate_questions(chat_client, config, num_questions, logger)
        elif llm_choice == '2':
            # Use Gemini-1.5-pro for persona generation
            questions = persona_strategy.generate_questions(gemini_evaluator, config, num_questions, logger)
        else:
            logger.error("Invalid LLM choice. Exiting the test.")
            return

    if not questions:
        logger.error("Generated questions are None or empty. Exiting the test.")
        return

    # Determine the test mode based on the strategy
    test_mode = persona_strategy.__class__.__name__.replace('PersonaStrategy', '').lower()

    processed_question_chunks = read_processed_chunks(source_dir)
    question_results = process_questions(chat_client, embedding_client, config, questions, processed_question_chunks, logger, llm_choice)
    save_results(test_destination_dir, question_results, test_mode)