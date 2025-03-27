import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from datasets import load_dataset
import numpy as np
import torch
import pandas as pd
import os
import subprocess
from dotenv import load_dotenv

# Function to load model, tokenizer, and dataset only once
@st.cache_resource  # This decorator caches the loading process
def load_resources():
    # load_dotenv()

    # Set the environment variable using os
    huggingface_token = st.secrets["huggingface"]["token"]

    # Run the huggingface-cli login command from the Python script using subprocess
    subprocess.run(["huggingface-cli", "login", "--token", huggingface_token])
    # Load the pre-trained Llama3 model (or your fine-tuned model)
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Replace with your fine-tuned model path or model name from Hugging Face
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config
    )

    # Load your dataset (this is just an example, replace with your actual dataset)
    dataset = load_dataset('qiaojin/PubMedQA', 'pqa_artificial')  # Replace with the Hugging Face dataset name
    
    # Dataset preprocessing (assuming the dataset is a Q&A dataset with columns 'question', 'answer', 'context')
    df = pd.DataFrame(dataset['train'])  # Replace 'train' with your dataset split
    df = df.sample(10000)  # Sample a subset of the dataset for faster processing
    # Embed the context (documents)
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedder = SentenceTransformer(embedding_model_name)

    # Create embeddings for the context
    document_embeddings = embedder.encode(df['context'].tolist(), convert_to_tensor=True)

    # Create FAISS index
    faiss_index = FAISS.from_embeddings(document_embeddings)

    # Create the retrieval chain with LangChain
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=faiss_index.as_retriever(),
        return_source_documents=True
    )
    
    # Return the loaded resources
    return model, tokenizer, dataset, faiss_index, qa_chain

# Load resources only once and cache them
if "model" not in st.session_state:
    with st.spinner("Loading the model and dataset... Please wait!"):
        model, tokenizer, dataset, faiss_index, qa_chain = load_resources()

        # Store the loaded resources in session state
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.dataset = dataset
        st.session_state.faiss_index = faiss_index
        st.session_state.qa_chain = qa_chain

# Initialize Streamlit session state for memory
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Streamlit UI Setup
st.title("Llama3-based Chatbot with RAG and Contextual Memory")
st.write("Ask a question, and I will provide two possible answers for you to choose from.")

# User input for the query
user_query = st.text_input("Your Query:", "")

if user_query:
    with st.spinner("Getting two possible answers..."):
        # Retrieve context using FAISS (document retrieval)
        result = st.session_state.qa_chain.run(user_query)
        
        # Show two possible responses (you can adjust this to get two different answers)
        possible_answers = [
            result['result'],  # First possible answer from model
            result['result']   # Duplicate for simplicity - in real cases, generate two variations (use different methods or prompt variations)
        ]
        
        # Save the current user query and model responses to the conversation history
        st.session_state.conversation_history.append({"question": user_query, "answers": possible_answers})
        
        # Show two responses for the user to choose from
        st.write("Possible Answers:")
        answer_1 = st.button("Answer 1: " + possible_answers[0], key="answer_1")
        answer_2 = st.button("Answer 2: " + possible_answers[1], key="answer_2")

        if answer_1:
            st.write(f"You selected Answer 1: {possible_answers[0]}")
            # Continue the conversation with the selected answer
            st.session_state.conversation_history.append({"question": "Answer 1 selected", "answers": [possible_answers[0]]})
        
        if answer_2:
            st.write(f"You selected Answer 2: {possible_answers[1]}")
            # Continue the conversation with the selected answer
            st.session_state.conversation_history.append({"question": "Answer 2 selected", "answers": [possible_answers[1]]})
        
        # Show the conversation history (for debug purposes, can be removed in production)
        st.write("Conversation History:")
        for conversation in st.session_state.conversation_history:
            st.write(f"Question: {conversation['question']}")
            st.write(f"Answers: {conversation['answers']}")
