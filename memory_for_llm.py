from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



#load data
DATA_PATH="data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,glob='*pdf',loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(data=DATA_PATH)
# print("len of document: ",len(documents))

# create chunks
def split_text(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunk = text_splitter.split_documents(extracted_data)
    return text_chunk

text_chunk = split_text(documents)
# print("len of chunks: ",len(text_chunk))

# embedding model
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()

# vectorstore
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunk,embedding_model)
db.save_local(DB_FAISS_PATH)