import os
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Function to grab all text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to grab text from CSV files
def get_csv_text(csv_docs):
    text = ""
    for csv_file in csv_docs:
        df = pd.read_csv(csv_file)
        text += df.to_string(index=False)  # Convert the DataFrame to a string format
    return text


# Function to slice the text into digestible chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # Returns a list of smaller text pieces


# Generate embeddings for each chunk of text
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# Create the chatbot logic for answering questions
def get_conversational_chain():
    prompt_template = """
    You are an AI assistant that will mimic the tone of the text as in the data. Respond conversationally, and if the answer is available in the document, make sure to reference the document name.

    Context: {context}?
    Question: {question}

    Answer in a conversational tone, including document references:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


# Clear previous chat history
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Okay, hit me with something now."}
    ]


# Handle user input
def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
        new_db = FAISS.load_local(
            "faiss_index", embeddings, allow_dangerous_deserialization=True
        )
        docs = new_db.similarity_search_with_score(user_question)  # Added scoring

        # Sort by relevance and keep the top 3 most relevant docs
        docs = sorted(docs, key=lambda x: x[1], reverse=True)[:3]

        if not docs:
            return {
                "output_text": [
                    "I could not find relevant information in the documents."
                ],
                "docs": docs,
            }

        chain = get_conversational_chain()

        # Prepare documents for context and chain processing
        response = chain(
            {"input_documents": [doc[0] for doc in docs], "question": user_question},
            return_only_outputs=True,
        )

        if response is None:
            raise RuntimeError("No response received from the model.")

        # Assuming response is a dictionary or string, handle it as a list for consistent formatting
        if isinstance(response, str):
            response = [response]

        return {"output_text": response, "docs": docs}

    except Exception as e:
        print(f"Error during processing: {e}")
        return {
            "output_text": [
                "I'm sorry, I encountered an issue processing your request."
            ],
            "docs": [],
        }


def main():
    st.set_page_config(page_title="ASR Upgraded", page_icon="💬")

    # Sidebar for PDF and CSV upload
    with st.sidebar:
        st.title("Control Panel")
        pdf_docs = st.file_uploader(
            "Drop your PDFs here", accept_multiple_files=True, type=["pdf"]
        )
        csv_docs = st.file_uploader(
            "Drop your CSV files here", accept_multiple_files=True, type=["csv"]
        )

        if st.button("Submit & Process"):
            if pdf_docs or csv_docs:
                with st.spinner("Processing documents... Please wait!"):
                    raw_text = ""
                    if pdf_docs:
                        raw_text += get_pdf_text(pdf_docs)
                    if csv_docs:
                        raw_text += get_csv_text(csv_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("All set! You can start chatting now.")
            else:
                st.error("Please upload at least one file (PDF or CSV).")

    # Main content area for chatbot interaction
    st.title("ASR Assistant")
    st.write("Let's talk! Upload some PDFs or CSVs and challenge me on it.")
    st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

    # Initialize chat session
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Okay, hit me with something now."}
        ]

    # Display existing chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input and bot response
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Process user input and generate response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_data = user_input(prompt)
                response = response_data.get("output_text")
                docs = response_data.get("docs", [])
                placeholder = st.empty()
                full_response = ""

                # Ensure response is a list of strings
                if isinstance(response, list):
                    full_response = " ".join(response)
                else:
                    full_response = str(response)

                # Check if docs is empty or has the expected structure
                if docs and hasattr(docs[0][0], "metadata"):
                    source = docs[0][0].metadata.get("source", "Unknown Source")
                else:
                    source = "No relevant documents found"

                # Embed document source into the response
                full_response += f"\n\n*Source: {source}*"
                placeholder.markdown(full_response)

        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
