import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Setup LLm (Mistral with huggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")
# HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
# HUGGINGFACE_REPO_ID="microsoft/Phi-3-mini-4k-instruct"
# HUGGINGFACE_REPO_ID = "google/gemma-7b-it"
HUGGINGFACE_REPO_ID= "mistralai/Mixtral-8x7B-Instruct-v0.1"


def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"tokens":HF_TOKEN,
                      "max_length":"512"} ,
        task="text-generation"        
    )
    return llm

# Connect LLM with FAISS and create chain

CUSTOM_PROMPT_TEMPLATE="""
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

context: {context}
Question: {question}

start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context","question"])
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH,embedding_model, allow_dangerous_deserialization=True)

#create QA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID) ,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':1}),
    return_source_documents=True ,
    chain_type_kwargs={"prompt":set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

#NOW INVOKE with A SINGLE QUERY

user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ",response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])
