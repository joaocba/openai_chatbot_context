from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import os


# set openai API key
# os.environ["OPENAI_API_KEY"] = ''
# in case it is already defined on windows path variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")


def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 250
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600
    # set dataset file name
    dataset_file_name = 'indices\index.json'

    # define LLM (Large Language Model)
    llm_predictor = LLMPredictor(
        llm=OpenAI(
            temperature = 0,
            model_name = "text-davinci-003",
            max_tokens = num_outputs
        )
    )

    prompt_helper = PromptHelper(
        max_input_size,
        num_outputs,
        max_chunk_overlap,
        chunk_size_limit = chunk_size_limit
    )

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(
        documents,
        llm_predictor = llm_predictor,
        prompt_helper = prompt_helper
    )

    index.save_to_disk(dataset_file_name)
    return index


def generate_response(prompt, index):
    response = index.query(prompt, response_mode="compact")
    return response.response