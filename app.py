# Define local llm
from langchain_ollama import ChatOllama

# use model name running in ollama locally
llm = ChatOllama(model="llama3.2")

# Define embeddings to convert documents to vectors
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings( model="llama3.2",)

# Define vector store to index documents
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

# Define document loader to load PDFs
from langchain_community.document_loaders import PyPDFDirectoryLoader

docs = []

loader = PyPDFDirectoryLoader("data/")
docs_lazy = loader.lazy_load()
for doc in docs_lazy:
    docs.append(doc)

# Index documents
vector_store = InMemoryVectorStore.from_documents(docs,embeddings)

# Define prompt for question-answering
from langchain import hub

prompt = hub.pull("rlm/rag-prompt")

# You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
# Question: {question} 
# Context: {context} 
# Answer:

from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
# Retrieve top documents close to the question as context
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

# Generate answer using retrieved documents in context
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Setup graph (retrieve -> generate)
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Define chainlit app / chatbot
import chainlit as cl
from langchain.chains import LLMChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain_core.messages import HumanMessage
from langchain.schema.runnable.config import RunnableConfig


@cl.on_chat_start
def quey_llm():
    conversation_memory = ConversationBufferMemory(memory_key="chat_history",
                                                   max_len=50,
                                                   return_messages=True,
                                                   )
    llm_chain = LLMChain(llm=llm, prompt=prompt, memory=conversation_memory)

    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")
    
    for msg, metadata in graph.stream({"question": msg.content}, stream_mode="messages", config=RunnableConfig(callbacks=[cb], **config)):
        if (
            msg.content
            and not isinstance(msg, HumanMessage)
            and metadata["langgraph_node"] == "generate"
        ):
            await final_answer.stream_token(msg.content)

    await final_answer.send()

# invoke graph by passing question (required by retrieve step)
# response = graph.invoke({"question": "What does response code of 123 mean?"})
# print(response["answer"])
