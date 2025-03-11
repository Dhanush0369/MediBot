import os
import streamlit as st

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DB_FAISS_PATH = "vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH,embedding_model, allow_dangerous_deserialization=True)
    return db


def load_llm(huggingface_repo_id, HF_TOKEN):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        Temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"512"}
    )
    
    return llm


def set_custom_promt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context","question"])
    return prompt


def main():
    st.title("Ask MediBot!")
    
    if 'messages' not in st.session_state:
        st.session_state.messages =[]
        
    
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    
    prompt = st.chat_input("Enter your question here")
    
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user','content':prompt})
        
        HF_TOKEN=os.environ.get("HF_TOKEN")
        HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
        
        CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """      
        
        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load vector data")
                
            qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID,HF_TOKEN=HF_TOKEN),
            chain_type="stuff",
            retriever=get_vectorstore().as_retriever(search_kwargs={'k':3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt':set_custom_promt(CUSTOM_PROMPT_TEMPLATE)} 
            )
            
            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role':'assistant','content':result})
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
        
     

if __name__ == "__main__":
    main()
    
    