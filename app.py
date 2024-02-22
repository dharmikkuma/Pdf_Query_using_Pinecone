from src.Project.utils.common import download_hf_embedding_model, query_from_existing_embeddings, qa_chain 
from flask import Flask, render_template, request
from transformers import pipeline
from dotenv import load_dotenv
import os
import pinecone


app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
PINECONE_HOST_URL = os.environ.get('PINECONE_HOST_URL')
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, host = PINECONE_HOST_URL)

embeddings_model = download_hf_embedding_model("sentence-transformers/all-MiniLM-L6-v2")

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    text = request.form["msg"]
    index = pc.Index("medicalbot", host = PINECONE_HOST_URL)
    results = query_from_existing_embeddings(str(text), embeddings_model, index, num_results=2)
    # answer = qa_chain(text, results, qa_pipeline)

    if results.matches:
        pages =[]
        scores = []
        for i in range(len(results.matches)):
            pages.append(results.matches[i].metadata["page"])
            scores.append(results.matches[i].score)
    else:
        pages = "N/A"
        scores = "N/A"
    # Construct the response as a JSON object
    response_data = {
        "answer": str(results.matches[0].metadata["text"]),
        "answer2": str(results.matches[1].metadata["text"]),
        "page_number": str(pages),
        "score": str(scores)
    }

    # Return the response as JSON
    return f"Answer is: {response_data['answer']}<br> second answer is:{response_data['answer2']}<br>from pages: {response_data['page_number']}<br>with scores {response_data['score']}"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)



