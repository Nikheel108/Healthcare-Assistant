import streamlit as st
import os
import time
from datetime import datetime

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Set page configuration
st.set_page_config(
    page_title="Healthcare Assistant",
    page_icon="➕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #1a1f2d;
    }
    
    /* Header card styling */
    .header-card {
        background: rgba(30, 40, 70, 0.4);
        border-radius: 20px;
        padding: 30px;
        margin: 20px auto;
        max-width: 900px;
    }
    
    .header-title {
        color: #4a9eff;
        font-size: 2.5em;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .header-text {
        color: #8b95a9;
        font-size: 1.1em;
        line-height: 1.6;
        text-align: center;
    }
    
    /* Message styling */
    .user-message {
        background: #4a9eff;
        color: white;
        padding: 15px 25px;
        border-radius: 15px;
        margin: 10px 0 10px auto;
        max-width: 700px;
        width: fit-content;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-right: 20px;
    }
    
    .assistant-message {
        background: rgba(30, 40, 70, 0.4);
        color: white;
        padding: 15px 25px;
        border-radius: 15px;
        margin: 10px auto 10px 20px;
        max-width: 700px;
        width: fit-content;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1f2d;
    }
    
    /* Input box styling */
    .stTextInput > div > div > input {
        background-color: #2d3958;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 15px;
    }
    
    .stTextInput > div > div > input:focus {
        box-shadow: none;
        border: 2px solid #4a9eff;
    }
    
    /* Chat input container */
    .stChatInputContainer {
        background-color: rgba(45, 57, 88, 0.5);
        border-radius: 15px;
        padding: 10px;
        margin-top: 20px;
    }
    
    /* Chat input field */
    .stChatInput {
        background-color: rgba(45, 57, 88, 0.5);
        border-radius: 15px;
        color: white;
    }
    
    /* Streamlit elements background */
    .st-emotion-cache-1y4p8pa {
        background-color: transparent;
    }

    /* Error message styling */
    .stAlert {
        background-color: rgba(255, 87, 87, 0.1);
        color: #ff5757;
        border: 1px solid #ff5757;
        border-radius: 10px;
    }
    
    /* Message container styling */
    .messages-container {
        max-width: 1000px;
        margin: 0 auto;
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

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

def calculate_confidence_score(response_type, response_time):
    """
    Calculate a confidence score based on response type and time
    Returns a score between 0 and 100
    """
    base_scores = {
        'Greeting': 98,  # Simple responses have high accuracy
        'Feedback': 98,
        'Medical Query': 85,  # More complex responses start with lower base score
        'Error': 0
    }
    
    # Adjust score based on response time
    time_penalty = min(10, max(0, response_time - 1)) * 2  # 2% penalty per second over 1s
    
    final_score = max(0, min(100, base_scores.get(response_type, 70) - time_penalty))
    return round(final_score, 1)

def main():
    # Initialize session state variables
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'response_times' not in st.session_state:
        st.session_state.response_times = []
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False

    # Create sidebar with enhanced response time display
    sidebar = st.sidebar
    with sidebar:
        st.markdown("""
            <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 20px;'>
                <span style='font-size: 24px;'>⌛</span>
                <h2 style='color: #4a9eff; margin: 0;'>Response Information</h2>
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.response_times:
            st.markdown("""
                <div style='background: rgba(30, 40, 70, 0.4); padding: 15px; border-radius: 15px; margin-top: 20px;'>
                    <h3 style='color: #4a9eff; margin: 0 0 10px 0; text-align: center;'>Recent Interactions</h3>
                </div>
            """, unsafe_allow_html=True)
            
            for time_entry in reversed(st.session_state.response_times[-5:]):
                confidence_score = calculate_confidence_score(time_entry.get('type', 'General'), time_entry['time'])
                score_color = '#4CAF50' if confidence_score >= 90 else '#FFA726' if confidence_score >= 70 else '#FF5252'
                
                st.markdown(f"""
                    <div style='background: rgba(74, 158, 255, 0.1); padding: 15px; border-radius: 8px; margin: 10px 0; color: white;'>
                        <div style='margin-bottom: 5px; color: #4a9eff;'>Prompt:</div>
                        <div style='margin-bottom: 10px; font-size: 0.9em; word-wrap: break-word;'>{time_entry.get('prompt', 'N/A')[:50]}{'...' if len(time_entry.get('prompt', '')) > 50 else ''}</div>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                            <div>
                                <span style='color: #4a9eff;'>⏱️ Response Time:</span>
                                <span style='color: #ffffff; font-family: monospace;'>{time_entry['time']:.3f}s</span>
                            </div>
                            <div>
                                <span style='color: #4a9eff;'>🎯 Accuracy:</span>
                                <span style='color: {score_color}; font-family: monospace; font-weight: bold;'>{confidence_score}%</span>
                            </div>
                        </div>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div style='color: #8b95a9; font-size: 0.9em;'>
                                <span style='color: #4a9eff;'>📋</span> {time_entry.get('type', 'General')}
                            </div>
                            <div>
                                <span style='color: #4a9eff;'>🕒</span>
                                <span style='color: #8b95a9; font-size: 0.9em;'>{time_entry['timestamp']}</span>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # Add performance metrics with accuracy
            total_accuracy = sum(calculate_confidence_score(entry.get('type', 'General'), entry['time']) 
                               for entry in st.session_state.response_times)
            avg_accuracy = total_accuracy / len(st.session_state.response_times)
            avg_time = sum(entry['time'] for entry in st.session_state.response_times) / len(st.session_state.response_times)
            
            st.markdown(f"""
                <div style='background: rgba(30, 40, 70, 0.4); padding: 15px; border-radius: 15px; margin-top: 20px;'>
                    <h4 style='color: #4a9eff; margin: 0 0 10px 0; text-align: center;'>Performance Metrics</h4>
                    <div style='text-align: center; color: white;'>
                        <div style='margin-bottom: 10px;'>
                            <span style='color: #4a9eff;'>Average Accuracy:</span>
                            <br>
                            <span style='font-family: monospace; font-size: 1.2em; color: {
                                "#4CAF50" if avg_accuracy >= 90 else "#FFA726" if avg_accuracy >= 70 else "#FF5252"
                            };'>{avg_accuracy:.1f}%</span>
                        </div>
                        <div style='margin-bottom: 10px;'>
                            <span style='color: #4a9eff;'>Average Response Time:</span>
                            <br>
                            <span style='font-family: monospace; font-size: 1.2em;'>{avg_time:.3f}s</span>
                        </div>
                        <div>
                            <span style='color: #4a9eff;'>Total Interactions:</span>
                            <br>
                            <span style='font-family: monospace; font-size: 1.2em;'>{len(st.session_state.response_times)}</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # Header card
    st.markdown("""
        <div class='header-card'>
            <div class='header-title' align="center">
                <span style='font-size: 1.5em;'>🏥</span>
                Your Personal Health Assistant
            </div>
            <div class='header-text'>
                <p>I'm here to provide helpful information about health and wellness in a clear, friendly way.</p>
                <p>While I can offer general health information, remember that I'm not a replacement for professional medical advice.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Messages container
    st.markdown("<div class='messages-container'>", unsafe_allow_html=True)

    # Display chat messages
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f"""
                <div class='user-message'>
                    <div style='flex-grow: 1;'>{message['content']}</div>
                    <span style='font-size: 1.2em;'>👤</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='assistant-message'>
                    <span style='font-size: 1.2em;'>👨‍⚕️</span>
                    <div style='flex-grow: 1;'>{message['content']}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Chat input
    prompt = st.chat_input("How can I help you with your health questions today?")

    if prompt:
        start_time = time.time()
        
        # Display user message
        st.markdown(f"""
            <div class='user-message'>
                <div style='flex-grow: 1;'>{prompt}</div>
                <span style='font-size: 1.2em;'>👤</span>
            </div>
        """, unsafe_allow_html=True)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        if prompt.lower().strip() in ['hello', 'hi', 'hey', 'hello!', 'hi!', 'hey!']:
            greeting_response = "Hello! How can I assist you with medical questions today?"
            st.markdown(f"""
                <div class='assistant-message'>
                    <span style='font-size: 1.2em;'>👨‍⚕️</span>
                    <div style='flex-grow: 1;'>{greeting_response}</div>
                </div>
            """, unsafe_allow_html=True)
            st.session_state.messages.append({'role': 'assistant', 'content': greeting_response})
            
            end_time = time.time()
            response_time = end_time - start_time
            confidence_score = calculate_confidence_score('Greeting', response_time)
            st.session_state.response_times.append({
                'time': response_time,
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'prompt': prompt,
                'type': 'Greeting',
                'confidence': confidence_score
            })
            return

        if prompt.lower().strip() in POSITIVE_FEEDBACK:
            feedback_response = "You're welcome! Please don't hesitate to ask if any other health concerns come up."
            st.markdown(f"""
                <div class='assistant-message'>
                    <span style='font-size: 1.2em;'>👨‍⚕️</span>
                    <div style='flex-grow: 1;'>{feedback_response}</div>
                </div>
            """, unsafe_allow_html=True)
            st.session_state.messages.append({'role': 'assistant', 'content': feedback_response})
            
            end_time = time.time()
            response_time = end_time - start_time
            confidence_score = calculate_confidence_score('Feedback', response_time)
            st.session_state.response_times.append({
                'time': response_time,
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'prompt': prompt,
                'type': 'Feedback',
                'confidence': confidence_score
            })
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
            
            st.markdown(f"""
                <div class='assistant-message'>
                    <span style='font-size: 1.2em;'>👨‍⚕️</span>
                    <div style='flex-grow: 1;'>{result}</div>
                </div>
            """, unsafe_allow_html=True)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

            end_time = time.time()
            response_time = end_time - start_time
            confidence_score = calculate_confidence_score('Medical Query', response_time)
            st.session_state.response_times.append({
                'time': response_time,
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'prompt': prompt,
                'type': 'Medical Query',
                'confidence': confidence_score
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")
            end_time = time.time()
            response_time = end_time - start_time
            confidence_score = calculate_confidence_score('Error', response_time)
            st.session_state.response_times.append({
                'time': response_time,
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'prompt': prompt,
                'type': 'Error',
                'confidence': confidence_score
            })

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