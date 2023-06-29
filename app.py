from flask import Flask, render_template, request
from gpt_index import GPTSimpleVectorIndex
from qa_engine import construct_index, generate_response


## context data config ##
# add data files to \content folder, can be .html, .txt, .csv, etc
content_fpath = 'content'

# construct and save index - this consumes usage on API credit
#index = construct_index(content_fpath) #uncomment to generate json

# path for dataset file (after indexation)
dataset_file = 'indices\index.json'
## end context data config ##


# array to store conversations
conversation = ["You are a virtual assistant and you speak portuguese."]    # define initial role

app = Flask(__name__)

# define app routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    # load indexed data
    index = GPTSimpleVectorIndex.load_from_disk(dataset_file)
    user_input = request.args.get("msg") + '\n'
    if user_input:
        conversation.append(f"{user_input}")

        # get conversation history
        prompt = "\n".join(conversation[-3:])

        # generate AI response based on indexed data
        response = generate_response(prompt, index)

        # add AI response to conversation
        conversation.append(f"{response}")

        return response
    else:
        return "Sorry, I didn't understand that."

if __name__ == "__main__":
    app.run()