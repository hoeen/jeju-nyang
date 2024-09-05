import bs4
import os

from langchain import hub
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# load vectorstore
vectorstore = Chroma(persist_directory="visitjeju_db_v3", embedding_function=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
# prompt = {'...'}
breakpoint()


def format_docs(docs):
    return "\n\n".join(doc.metadata['description'] for doc in docs)


template = """
    You are a Korean assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer: 
"""

prompt = ChatPromptTemplate.from_template(template)

# prompt = hub.pull("rlm/rag-prompt")
# prompt link : https://smith.langchain.com/hub/rlm/rag-prompt

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# qa_chain = RetrievalQA.from_chain_type(
#     llm, retriever=vectorstore.as_retriever(), chain_type_kwargs={"prompt": prompt}
# )

# result = qa_chain.run("성산일출봉이 천연기념물로 지정된 날짜는?")
# print(result)


# breakpoint()
while True:
    question = input("Question: ")
    if question == 'break':
        break
    print(rag_chain.invoke(question))

# print(rag_chain.invoke("사려니숲길 탐방 방법에 대해 설명해줘"), '\n', '='*20)
# print(rag_chain.invoke("카멜리아힐의 대략적 관람 시간은?"), '\n', '='*20)
# # print(rag_chain.invoke("우도에 대해 설명해줘"), '\n', '='*20)
# # print(rag_chain.invoke("성산일출봉에 대해 설명해줘"), '\n', '='*20)


# # print(rag_chain.invoke("성산일출봉의 아픈 역사에 대해 설명해줘"), '\n', '='*20)


# print(rag_chain.invoke("성산일출봉이 천연기념물로 지정된 날짜는?"), '\n', '='*20)
