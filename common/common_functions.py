# Standard library imports
import os
from openai import AzureOpenAI
import logging

from common.ApiConfiguration import ApiConfiguration
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type
from openai import BadRequestError

config = ApiConfiguration()

def ensure_directory_exists(directory):
    """
    Checks if the directory at the given destination exists.
    If it does not exist, creates the directory.

    Parameters:
    directory (str): The path to the directory.
    """
    # Use os.path.join() to handle path construction across different platforms
    if not os.path.exists(directory):
        os.makedirs(directory)
        #print(f"Directory '{directory}' created.")  #  can remove or comment out this print statement for production
    else:
        # print(f"Directory '{directory}' already exists.")
        pass

# Construct the path using os.path.join() for cross-platform compatibility
HTML_DESTINATION_DIR = os.path.join("data", "web")
ensure_directory_exists(HTML_DESTINATION_DIR)

def get_embedding(text: str, embedding_client: AzureOpenAI, config: ApiConfiguration, model: str = "text-embedding-3-large"):
    # Replace newlines with spaces 
    text = text.replace("\n", " ")

    # Use the provided model parameter if given, otherwise fall back to config's deployment name
    chosen_model = model if model else config.embedModelName

    # Generate embedding using the chosen model and configuration
    response = embedding_client.embeddings.create(
        input=[text],
        model=chosen_model,
        timeout=config.openAiRequestTimeout
    )
    
    return response.data[0].embedding

# MAX_RETRIES = 15                    # Maximum number of retries for API calls
# NUM_QUESTIONS = 100   

# # Function to call the OpenAI API with retry logic
# @retry(wait=wait_random_exponential(min=5, max=15), stop=stop_after_attempt(MAX_RETRIES), retry=retry_if_not_exception_type(BadRequestError))
# def call_openai_chat(chat_client: AzureOpenAI, messages: list[dict[str, str]], config: ApiConfiguration, logger: logging.Logger) -> str:
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
