# -*- coding: utf-8 -*-
from langchain_community.document_loaders import CSVLoader
# from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

loader = CSVLoader(file_path='./wifi_list_latitude_processed.csv',csv_args={'delimiter': ',','fieldnames': ['apGroupName','category','latitude','longitude']})
data = loader.load()

with open("./asd.txt", 'r') as f:
    api_key = f.read()
os.environ["OPENAI_API_KEY"] = api_key

# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size=50,
#     chunk_overlap=20,
#     length_function=len,
#     is_separator_regex=False,
# )

embedding_function = OpenAIEmbeddings()
db = Chroma.from_documents(data, embedding_function, persist_directory="wifilist_lat_db_v2")
retriever = db.as_retriever()


# print(retriever.invoke("기당미술관의 latitude, longitude를 고려했을 때, 가장 가까운 공원은 뭐야?"))