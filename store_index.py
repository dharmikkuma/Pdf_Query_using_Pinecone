from src.Project.utils.common import load_data, split_text, download_hf_embedding_model, save_embeddings
from dotenv import load_dotenv
import os
import pinecone


load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
PINECONE_HOST_URL = os.environ.get('PINECONE_HOST_URL')

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, host = PINECONE_HOST_URL)
index_name = "yout index name"
index = pc.Index(index_name, host = PINECONE_HOST_URL)

data = load_data('data/')
text_chunks = split_text(data)

embeddings_model = download_hf_embedding_model("sentence-transformers/all-MiniLM-L6-v2")

docsearch = save_embeddings(text_chunks, embeddings_model, index_name)










