from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone



def load_data(Path: str):
    
    loader = DirectoryLoader(Path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    return documents

def split_text(documents, chunk_size: int, chunk_overlap:int):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(documents)

    return text_chunks

def download_hf_embedding_model(model_name: str):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return embeddings

def save_embeddings(text_chunks, embeddings, index_name):
    docsearch = Pinecone.from_documents(text_chunks, embeddings, index_name=index_name)

    return docsearch

def query_from_existing_embeddings(text: str, embeddings_model, index, num_results=5):
    doc_result = embeddings_model.embed_documents([text])
    results = index.query(vector=doc_result[0], top_k=num_results, include_values=True, include_metadata=True)
    return results


def qa_chain(question, results, qa_pipeline):

    concatenated_text = ""
    for i in range(len(results.matches)):
        docs = results.matches[i].metadata["text"]
        concatenated_text += docs + " "
    answer = qa_pipeline(question=question, context=concatenated_text)
    return answer






