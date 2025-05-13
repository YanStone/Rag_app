from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda

from langchain_core.documents import Document
from typing_extensions import List, TypedDict, Annotated
from langgraph.graph import START, StateGraph, MessagesState

from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate

from langchain_core.messages import AIMessage, SystemMessage

from langchain_huggingface import HuggingFaceEndpoint

from dotenv import load_dotenv
import os


local_llm = False

load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


def load_and_split(path):
    loader = PyMuPDFLoader(path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, add_start_index=True)
    chunks = splitter.split_documents(documents)
    return chunks

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def create_graph(vectorstore):
    # retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    # llm = Ollama(model="mistral")
    # qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    # return qa_chain

    # llm = OllamaLLM(model="mistral:instruct-q4")

    # if local_llm:
    #     llm = OllamaLLM(model="mistral")
    # else:
    llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=api_token)    )

    # prompt = hub.pull("rlm/rag-prompt")

    # prompt = ChatPromptTemplate.from_messages([
    # SystemMessagePromptTemplate.from_template(
    #     "You are a helpful assistant that answers questions based on context extracted from documents. "
    #     "Use only the context provided. If the answer cannot be found, say 'I don't know'."
    # ),
    # HumanMessagePromptTemplate.from_template("Context:\n{context}"),
    # MessagesPlaceholder(variable_name="messages"), ])

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are a helpful assistant that answers questions based on context extracted from documents. "
            "You will receive the previous messages (there might be none), the context and then the question. Answer directly with your answer. If the answer cannot be found, say 'I don't know'."
        )),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template(
            "Context:\n{context}\nQuestion: {question} \nAnswer:"
        ),
    ])



    # prompt = PromptTemplate.from_template(prompt_template)

    # def format_docs(docs):
    #     return "\n\n".join(doc.page_content for doc in docs)
    
    # def format_docs_for_prompt(inputs):
    #     docs = inputs["documents"]
    #     question = inputs["question"]
    #     context = "\n\n".join(doc.page_content for doc in docs)
    #     return {"context": context, "question": question, "documents": docs}

    # format_docs_runnable = RunnableLambda(format_docs_for_prompt)

    # def combine_answer_and_sources(inputs):
    #     return {
    #         "answer": inputs["answer"],
    #         "sources": inputs["documents"],
    #     }

    # final_output_runnable = RunnableLambda(combine_answer_and_sources)



    # qa_chain = (
    # {
    #     "documents": vectorstore.as_retriever(search_kwargs={"k": 3}),
    #     "question": RunnablePassthrough(),
    # }
    # | format_docs_runnable  # Formats docs into context
    # | prompt
    # | llm
    # | RunnableLambda(lambda x: {"answer": x["answer"], "documents": x["documents"]})  # Retain docs with answer
    # | final_output_runnable  # Return both
    # )

    class State(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        context: List[Document]
    
    def retrieve(state: State):
        retrieved_docs = vectorstore.similarity_search(state["messages"][-1].text())
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])

        
        # messages = prompt.invoke({"question": state["messages"][-1].text(), "context": docs_content})
        prompt_input = prompt.invoke({
                "messages": state["messages"][:-1],
                "question": state["messages"][-1].text(),
                "context": docs_content,
        })        
        response = llm.invoke(prompt_input)
        new_messages = state["messages"] + [AIMessage(content=response)]
        return {"messages": new_messages, "context": state["context"]}

    
    
    
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    return graph


# TODO

#remove beginning of page if they are the same (only keep one)

#add memory to the chat, like chatgpt (cf doc langchain)

#clear question box when uploading new file

#for multiple different sources :
#what is one ethical issue concerning fine-tuned language models?
