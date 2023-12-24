import os
import streamlit as st

from pathlib import Path

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

# -- Helpers -- #
def format_document(doc, index, prompt):
    """Format a document into a string based on a prompt template."""
    # Create a dictionary with document content and metadata.
    base_info = {"page_content": doc.page_content, "index": index, "source": doc.metadata['source']}
    
    # Check if any metadata is missing.
    missing_metadata = set(prompt.input_variables).difference(base_info)
    if len(missing_metadata) > 0:
        raise ValueError(f"Missing metadata: {list(missing_metadata)}.")
    
    # Filter only necessary variables for the prompt.
    document_info = {k: base_info[k] for k in prompt.input_variables}
    return prompt.format(**document_info)

# -- Prompts -- #
SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an English Teacher and you have to create a conversation and correct the user message if needed. Respond with no more than 100 words.
                Sources:
                {context}

                Question:
                {question}""",
)

# -- Custom Chains -- #
class StuffDocumentsWithIndexChain(StuffDocumentsChain):
    """Custom chain class to handle document combination with source indices."""
    def _get_inputs(self, docs, **kwargs):
        # Format each document and combine them.
        doc_strings = [
            format_document(doc, i, self.document_prompt)
            for i, doc in enumerate(docs, 1)
        ]
        
        # Filter only relevant input variables for the LLM chain prompt.
        inputs = {k: v for k, v in kwargs.items() if k in self.llm_chain.prompt.input_variables}
        inputs[self.document_variable_name] = self.document_separator.join(doc_strings)
        return inputs

# -- Functions -- #
@st.cache_resource(show_spinner=False)
def load_initial_resources():
    root = Path(__file__).absolute().parent / 'pdf_data'
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_APIKEY'])
    pdf_path = None
    if pdf_path is not None:
        data = load_pdf_data(pdf_path, text_splitter)
        doc_search = Chroma.from_documents(data, embeddings, collection_name='my_collection')
    else:
        data = []
        doc_search = None
    return {
        'root': root,
        'data': data,
        'vector_database': doc_search,
        'text_splitter': text_splitter,
        'embeddings': embeddings,
        'chain': load_qa_with_references()
    }

def load_qa_with_references():
    # Initialize the custom chain with a specific document format.
    llm = ChatOpenAI(model='gpt-4-1106-preview', temperature=0.0, openai_api_key=os.environ['OPENAI_APIKEY'])
    return StuffDocumentsWithIndexChain(
        llm_chain=LLMChain(
            llm=llm,
            prompt=SYSTEM_PROMPT,
        ),
        document_prompt=PromptTemplate(
            input_variables=["index", "source", "page_content"],
            template="[{index}] {source}:\n{page_content}",
        ),
        document_variable_name="context",
    )

def load_pdf_data(filename, splitter):
    documents = UnstructuredPDFLoader(str(filename), strategy='fast', chunking_strategy="by_title").load()
    return splitter.split_documents(documents)

def add_to_chat_history(message, role, print_to_chat=True):
    st.session_state.messages.append({'role': role, 'content': message})
    if print_to_chat:
        st.chat_message(role).markdown(message)

def answering_question(question, response_to_not_knowing_the_answer):
    # -- Find documents related to the question -- #
    with st.spinner('Generating response from vector DB...'):
        try:
            related_documets = st.session_state.resources['vector_database'].similarity_search(question)
        except Exception:
            related_documets = []

    # -- Answering -- #
    with st.spinner('Answering...'):
        answer =  st.session_state.resources['chain'].run(input_documents=related_documets, question=question)
    return answer

def delete_cache():
    st.session_state.resources['vector_database'].delete_collection()
    st.session_state.resources['vector_database'] = Chroma.from_documents(
       st.session_state.resources['data'],
       st.session_state.resources['embeddings'],
       collection_name='my_collection'
    )