from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_classic.chains import RetrievalQA

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt_tab')


docs = "/Users/abhishekkumar/Documents/ai-codes/chat-botes/docs"
vector_db_path = "/Users/abhishekkumar/Documents/ai-codes/chat-botes/vector_db"
collection_name = "document_collection"


embedding = HuggingFaceEmbeddings()

loader = DirectoryLoader(path="docs",glob="./*.pdf",loader_cls=UnstructuredFileLoader)
documents = loader.load()

# print(type(documents))
# print(len(documents))
# print(documents[0])

text_splitter = CharacterTextSplitter(chunk_size=2000,chunk_overlap=500)
text_chunks = text_splitter.split_documents(documents)

# creating the vector store
vector_store = Chroma.from_documents(documents=text_chunks,
                                     embedding=embedding,
                                     persist_directory=vector_db_path,
                                     collection_name=collection_name)


# load_dotenv()
# retrive

llm = ChatOllama(
    model="gemma3:4b",
    temperature=0.0
)
vector_data=Chroma(collection_name=collection_name,
                   embedding_function=embedding,
                   persist_directory=vector_db_path)


retriever = vector_data.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

query="what is vajra akeyless ?"
response = qa_chain.invoke({"query": query})
print(response["result"])


