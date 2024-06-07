from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import sys

load_dotenv()  # Take environment variables from .env (especially the Google API key)

# Check if the Google API key is available
google_api_key = os.environ.get("GOOGLE_API_KEY")
if not google_api_key:
    print("Error: GOOGLE_API_KEY environment variable not found.")
    sys.exit(1)

# Create Google Palm LLM model
llm = GooglePalm(google_api_key=google_api_key, temperature=0.1)

# Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

def create_vector_db():
    try:
        # Load data from FAQ sheet
        loader = CSVLoader(file_path='data1.csv', source_column="Question", encoding='utf-8')
        data = loader.load()

        # Create a FAISS instance for vector database from 'data'
        vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)

        # Save vector database locally
        vectordb.save_local(vectordb_file_path)
        print("Vector database created and saved locally.")

    except Exception as e:
        print(f"An error occurred while creating the vector database: {e}")

def get_qa_chain():
    try:
        # Load the vector database from the local folder
        vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

        # Create a retriever for querying the vector database
        retriever = vectordb.as_retriever(score_threshold=0.5)

        prompt_template = """Given the following context and a question, generate an answer based on this context only.
        Provide as much information from the "Answer" section in the source document as possible without making many changes.
        If the answer is not found in the context, kindly state "I don't know." Do not fabricate an answer.

        CONTEXT: {context}

        QUESTION: {question}"""

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain = RetrievalQA.from_chain_type(llm=llm,
                                            chain_type="stuff",
                                            retriever=retriever,
                                            input_key="query",
                                            return_source_documents=True,
                                            chain_type_kwargs={"prompt": PROMPT})

        return chain

    except Exception as e:
        print(f"An error occurred while getting the QA chain: {e}")

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    if chain:
        print(chain({"query": "Do you have a JavaScript course?"}))
