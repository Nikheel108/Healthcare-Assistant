import streamlit as st
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = "vectorstore/db_faiss"
POSITIVE_FEEDBACK = ['good', 'great', 'thanks', 'thank you', 'excellent', 'wonderful', 'nice']

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"tokens": HF_TOKEN,
                      "max_length": "512"},
        task="text-generation"
    )
    return llm

def main():
    st.title("Healthcare Assistant!!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        if prompt.lower().strip() in ['hello', 'hi', 'hey', 'hello!', 'hi!', 'hey!']:
            greeting_response = "Hello! How can I assist you with medical questions today?"
            st.chat_message('assistant').markdown(greeting_response)
            st.session_state.messages.append({'role': 'assistant', 'content': greeting_response})
            return  # Exit the function early

        if prompt.lower().strip() in POSITIVE_FEEDBACK:
            feedback_response = "You're welcome! Please don't hesitate to ask if any other health concerns come up."
            st.chat_message('assistant').markdown(feedback_response)
            st.session_state.messages.append({'role': 'assistant', 'content': feedback_response})
            return

        CUSTOM_PROMPT_TEMPLATE = """You are a friendly medical assistant. Provide helpful advice in simple, natural language that a patient would understand.
                                    Use the pieces of information provided in the context to answer user's question.
                                    If the context doesn't contain the answer, say "I don't have enough medical information about that."
                                    Keep responses professional but conversational.


                                    context: {context}
                                    Question: {question}

                                    start the answer directly. No small talk please.
                                    """

        HUGGINGFACE_REPO_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 1}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            source_documents = response["source_documents"]

            result_to_show = result

            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

# --- How to Run ---
# 1. Ensure you have a vector store named 'db_faiss' in a 'vectorstore' directory.
# 2. Set the HF_TOKEN environment variable with your Hugging Face API token.
# 3. Install necessary libraries: pip install streamlit langchain langchain-huggingface faiss-cpu sentence-transformers
# 4. Run the app: streamlit run your_script_name.py


# --- Pipenv Instructions (if needed) ---
# 1. Install pipenv: pip install pipenv
# 2. Create a Pipfile with the dependencies (streamlit, langchain, etc.)
# 3. Activate the environment: pipenv shell
# 4. Set HF_TOKEN environment variable within the shell if needed.
# 5. Run the app: streamlit run your_script_name.py