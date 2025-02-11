import os
import gc
import tempfile
import uuid
import pandas as pd
import streamlit as st

# Import required classes from llama_index modules
from llama_index.core import Settings, PromptTemplate, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.docling import DoclingReader
from llama_index.core.node_parser import MarkdownNodeParser

# -------------------------------------------------------------------
# Main App Background Image
# -------------------------------------------------------------------
def set_background_image():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://i.postimg.cc/R02gf8Ys/background-final.png");
            background-size: cover;
            background-position: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_image()

# -------------------------------------------------------------------
# Sidebar Background Image and Text Styling
# -------------------------------------------------------------------
sidebar_background_image = '''
<style>
[data-testid="stSidebar"] {
    background-image: url("https://i.postimg.cc/wB1W1HwR/sidebar-final.png");
    background-size: cover;
    background-position: center;
}
</style>
'''
st.sidebar.markdown(sidebar_background_image, unsafe_allow_html=True)

sidebar_text_css = """
<style>
.sidebar-header {
    color: black;
}
</style>
"""
st.sidebar.markdown(sidebar_text_css, unsafe_allow_html=True)

# -------------------------------------------------------------------
# Hide default Streamlit style elements (menu, header, footer)
# -------------------------------------------------------------------
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# -------------------------------------------------------------------
# Clear Button Custom CSS
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #ff5004;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------------
# Custom CSS to force file uploader text (including loaded file name) to be black
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    [data-baseweb="fileUploader"] * {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------------
# Session State Setup
# -------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.engine_dict = {}

query_handler = None

# -------------------------------------------------------------------
# Cached LLM Initialization
# -------------------------------------------------------------------
@st.cache_resource
def init_llm():
    return Ollama(model="llama3.2", request_timeout=120.0)

# -------------------------------------------------------------------
# Chat History Reset Function
# -------------------------------------------------------------------
def reset_conversation():
    st.session_state["chat_history"] = []
    st.session_state["extra_context"] = None
    gc.collect()

# -------------------------------------------------------------------
# Excel Display Function
# -------------------------------------------------------------------
def preview_excel(file_obj):
    st.markdown("### Excel File Preview")
    try:
        df = pd.read_excel(file_obj)
        st.dataframe(df)
    except Exception as err:
        st.error(f"Error reading Excel file: {err}")

# -------------------------------------------------------------------
# Sidebar: Document Upload and Indexing
# -------------------------------------------------------------------
with st.sidebar:
    # 1. Display the centered, smaller logo at the top of the sidebar.
    st.markdown(
        '<div style="text-align: center;">'
        '<img src="https://i.postimg.cc/2y00JJ1D/Microsoft-Excel-Logo.png" width="150">'
        '</div>',
        unsafe_allow_html=True
    )
    
    # 2. Display the social icons (GitHub and LinkedIn)
    social_icons_html = """
    <div style="text-align: center; margin-top: 10px; margin-bottom: 10px;">
        <a href="https://github.com" style="display: block; margin-bottom: 5px; text-decoration: none;">
            <img src="https://img.shields.io/badge/GitHub-Profile-181717?style=for-the-badge&logo=github&logoColor=white" 
                 alt="GitHub" style="width: 100px;" />
        </a>
        <a href="https://www.linkedin.com/in" style="display: block; margin-bottom: 5px; text-decoration: none;">
            <img src="https://img.shields.io/badge/LinkedIn-Profile-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" 
                 alt="LinkedIn" style="width: 100px;" />
        </a>
    </div>
    """
    st.markdown(social_icons_html, unsafe_allow_html=True)
    
    # 3. Display a bordered container with upload instructions.
    st.markdown(
        """
        <div style="border: 2px solid white; border-radius: 8px; padding: 10px; margin: 10px;">
            <div style="font-size: 12px; font-weight: bold; color: white; text-align: center;">
                Just drop your file, and let AI spill the tea ☕
            </div>
            <p style="color:black; text-align: center;"></p>
        </div>
        """,
        unsafe_allow_html=True
    )
    # 4. File uploader (loaded file name now appears in black)
    uploaded_excel = st.file_uploader("", type=["xlsx", "xls"])

# -------------------------------------------------------------------
# Main Content: Header & Clear Button (always at the top)
# -------------------------------------------------------------------
col_header, col_button = st.columns([6, 1])
with col_header:
    st.markdown(
        """
        <div style="border: 2px solid black; border-radius: 8px; padding: 10px; margin: 10px; 
                    background-color: #01ac64; text-align: center; min-height: 60px;
                    display: flex; align-items: center; justify-content: center;">
            <span style="font-size: 30px; font-weight: bold; color: white;">Your Excel Files Just Got Smarter 💡</span>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_button:
    st.markdown(
        "<div style='display: flex; align-items: center; justify-content: center; height: 100%;'>",
        unsafe_allow_html=True
    )
    st.button("Clear ↺", on_click=reset_conversation)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Document Processing & File Preview (renders below the header)
# -------------------------------------------------------------------
if uploaded_excel is not None:
    with st.spinner("Indexing your document. Please wait..."):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = os.path.join(temp_dir, uploaded_excel.name)
                with open(temp_file, "wb") as f_out:
                    f_out.write(uploaded_excel.getvalue())

                # Create a unique key for this file using the session id and file name
                file_key = f"{st.session_state.session_id}-{uploaded_excel.name}"

                if file_key not in st.session_state.engine_dict:
                    if os.path.exists(temp_dir):
                        extractor = DoclingReader()
                        folder_loader = SimpleDirectoryReader(
                            input_dir=temp_dir,
                            file_extractor={".xlsx": extractor},
                        )
                    else:
                        st.error("Could not locate the temporary directory.")
                        st.stop()

                    # Load documents from the folder
                    documents = folder_loader.load_data()

                    # Initialize LLM and Embedding model
                    llm_model = init_llm()
                    embedder = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
                    Settings.embed_model = embedder

                    # Create an index over the documents using a markdown node parser
                    node_processor = MarkdownNodeParser()
                    index_object = VectorStoreIndex.from_documents(
                        documents=documents, 
                        transformations=[node_processor],
                        show_progress=True
                    )

                    Settings.llm = llm_model
                    local_query_handler = index_object.as_query_engine(streaming=True)

                    # Customize the prompt template
                    custom_prompt = (
                        "Context information is provided below:\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Based on the above context, answer the query in a step-by-step, concise, and precise manner. If uncertain, reply with 'I don't know!'.\n"
                        "Query: {query_str}\n"
                        "Answer: "
                    )
                    prompt_obj = PromptTemplate(custom_prompt)
                    local_query_handler.update_prompts(
                        {"response_synthesizer:text_qa_template": prompt_obj}
                    )

                    st.session_state.engine_dict[file_key] = local_query_handler
                    query_handler = local_query_handler
                else:
                    query_handler = st.session_state.engine_dict[file_key]
        except Exception as ex:
            st.error(f"An error occurred: {ex}")
            st.stop()
    st.success("Your document has been processed. Chat is now ready!")
    preview_excel(uploaded_excel)

# -------------------------------------------------------------------
# Chat Interface (Below the header and file processing messages)
# -------------------------------------------------------------------
if "chat_history" not in st.session_state:
    reset_conversation()

for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

user_input = st.chat_input("Type your question: What’s the total revenue?")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_display = st.empty()
        complete_reply = ""
        if query_handler is None:
            st.error("No document has been processed yet. Please upload an Excel file.")
        else:
            stream_result = query_handler.query(user_input)
            for segment in stream_result.response_gen:
                complete_reply += segment
                response_display.markdown(complete_reply + "▌")
            response_display.markdown(complete_reply)
            st.session_state.chat_history.append({"role": "assistant", "content": complete_reply})
