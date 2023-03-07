from flask import Flask, render_template, request
from gpt_index import GPTSimpleVectorIndex
from qa_engine import construct_index, ask_qa


app = Flask(__name__)

# add data files to \content folder, can be .html, .txt, etc
content_fpath = 'content'

# construct and save GPT index (uncomment to index data) - this consumes usage on API credit
#index = construct_index(content_fpath)

# dataset file for context
dataset_file = 'indices\index.json'


conversations = []

# views
@app.route('/', methods=['GET', 'POST'])
def home():
    index = GPTSimpleVectorIndex.load_from_disk(dataset_file)
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        query = request.form['query']
        response = ask_qa(query, index)

        conversations.append(query)
        conversations.append(response)

        return render_template('index.html', chat = conversations)
    else:
        return render_template('index.html')


# run
if __name__ == '__app__':
    app.run()
