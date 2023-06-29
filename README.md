## Chatbot AI Assistant (Context Knownledge Base)

#### Technology: OpenAi, LlamaIndex, LangChain
#### Method: Completions (davinci model)

#### Description:
Chatbot developed with Python and Flask that features conversation with a virtual assistant. This uses a context based conversation and the answers are focused on a local indexation file structured with Llamaindex (GPT-index) lib which converts raw data into a vector dataset (Json) a therefore the assistant will use the dataset to provide answers. It allows to define an initial role and personification.

It make use of LlamaIndex (GPT Index) that provides a central interface to connect LLM’s (large language models) with external data.
On this demo is used GPTSimpleVectorIndex to created a vector file from the local documents.

LangChain is a framework built around LLMs. The core of the library is to “chain” together different components to create more advanced use cases around LLMs.

### How to run (commands Windows terminal with Python 2.7):

#### Part One: Prepare Environment
- **Define necessary parameters (OpenAi API key, ...) on file 'qa_engine.py'**
- Initialize virtual environment and install dependencies, run:

	    virtualenv env
	    env\Scripts\activate
	    pip install flask python-dotenv
        pip install openai
		pip install gpt_index langchain

#### Part Two: Prepare local content
- Add documents to folder "content"

#### Part Three: Run the app
- Initialize the app:

	    flask run

- Enter "http://localhost:5000" on browser to interact with app

#### Changelog
- v0.1
	- initial build