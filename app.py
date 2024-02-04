import streamlit as st
from llama_index.vector_stores import AstraDBVectorStore
from llama_index import (
    VectorStoreIndex,
    StorageContext,
)
from dotenv import load_dotenv
import os

# Load API secrets
load_dotenv()
LLAMA_PARSE_API_KEY = os.environ.get("LLAMA_PARSE_API_KEY")
ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Load index from Astra DB
def load_jobs_retriever(similarity_top_k=5):
    jobs_store = AstraDBVectorStore(
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        collection_name="jobo_jobs",
        embedding_dimension=1536,
    )
    index = VectorStoreIndex.from_vector_store(vector_store=jobs_store)
    retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    return retriever

# Load resume index from Astra DB
def load_resume_query_engine(collection_name):
    astra_db_store = AstraDBVectorStore(
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        collection_name=collection_name,
        embedding_dimension=1536,
    )
    index = VectorStoreIndex.from_vector_store(vector_store=astra_db_store)
    query_engine = index.as_query_engine()
    return query_engine

# Display retrivals
def display_retrival(retrival, scored=False):
    metadata = retrival.metadata
    loc = " - " + metadata["location"] if metadata["location"] else ""
    job_type = " - " + metadata["job_type"] if metadata["job_type"] else ""

    if scored:
        score = "    " + str(round(retrival.score * 100)) + "% match"
        card_html = (
            f'<a href="{metadata["job_url"]}" target="_blank" style="display: block; padding: 10px; border: 1px solid #ccc; margin: 10px 0; border-radius: 5px; text-decoration: none; color: white;">'
            f'<strong>{metadata["title"]}</strong><span style="color: #00ff00;">&nbsp;&nbsp;&nbsp;{score}</span>'
            f'<br> at <span style="color: #a57fc0;">{metadata["company"]} {loc} {job_type}</span>'
            '</a>'
        )
    else:
        card_html = (
            f'<a href="{metadata["job_url"]}" target="_blank" style="display: block; padding: 10px; border: 1px solid #ccc; margin: 10px 0; border-radius: 5px; text-decoration: none; color: white;">'
            f'<strong>{metadata["title"]}</strong>'
            f'<br> at <span style="color: #a57fc0;">{metadata["company"]} {loc} {job_type}</span>'
            '</a>'
        )

    st.markdown(card_html, unsafe_allow_html=True)


# Set up Streamlit app
def main():

    st.image("logo.png")

    # Initialize the state of app, so Jobo only responses after User has inputted something
    st.session_state.role = ["jobo"]

    # Load job index retriever
    jobs_retriever = load_jobs_retriever(similarity_top_k=5)

    # Load resume index
    resume_query_engine = load_resume_query_engine("bassim_resume")
    resume_response = resume_query_engine.query(
        "Summarize this person's education, work experiences, and skills in less than 100 words."
        ).response

    # User input: PDF file upload
    # For demo, always run this
    resume_summary = None
    if uploaded_file := st.file_uploader("Upload a PDF resume", type=["pdf"]):
        # Produce a summary of the resume and a query engine for the resume information if available
        # parsed_documents = llamaparse_text_from_pdf(uploaded_file)
        # resume_query_engine = create_resume_query_engine(parsed_documents, "jobo_resume")
        resume_summary = resume_response

    # User input: Keywords
    if keywords := st.text_input("Enter job search query:"):
        if resume_summary:
            retrivals = jobs_retriever.retrieve(keywords + " jobs that are good matches for a person with these qualifications: " + resume_summary)
        else:
            retrivals = jobs_retriever.retrieve(keywords)
        # Retrieve jobs based on semantic search based on resume summary
        # then do a second pass filtering using BM25 of the keywords
        # then a third pass of reranking for better relevance???
        # TODO: add a hybrid retriver with reranker
        # TODO: convert retrivals to table
        if len(retrivals) == 0:
            st.write("No results")
        else:
            for ret in retrivals:
                # st.write(ret) # for debugging
                display_retrival(ret, resume_summary is not None)

    st.image("power.png")

if __name__ == "__main__":
    main()