## DEMO: Q&A Assistant (Context Based)

#### Technology: OpenAi GPT AI models

#### Description: Simple chatbot demo developed with Flask that features conversation with a virtual assistant. The context of answers are focused on a local indexation file strutured with Llamaindex (GPT-index) lib which converts raw data into a dataset (Json) a therefore the assistant will use the dataset to provide answers. 

#### Place raw data on "content" folder

#### How to run (commands Windows terminal):
- python3 -m venv venv
- venv\Scripts\activate
- pip install flask openai python-dotenv gpt_index langchain
- flask run
- Load http://localhost:5000 on browser


#### Changelog
- v0.1
	- inital build
