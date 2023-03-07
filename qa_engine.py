from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import os


# set OpenAi API key ( grab it from https://platform.openai.com/account/api-keys )
os.environ["OPENAI_API_KEY"] = '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'


def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 150
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600
    # set dataset file name
    dataset_file_name = 'indices\index.json'

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(
        temperature=0, model_name="text-davinci-003", max_tokens=num_outputs))
    prompt_helper = PromptHelper(
        max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index.save_to_disk(dataset_file_name)

    return index


def ask_qa(query, index):
    response = index.query(query, response_mode="compact")
    return response.response