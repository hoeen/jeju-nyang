import os, time, re
from tqdm import tqdm

from selenium import webdriver

from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLSectionSplitter
from langchain_openai import ChatOpenAI
from langchain_community.document_transformers.html2text import Html2TextTransformer

from bs4 import BeautifulSoup


def title_cleaner(title: str) -> str:
    # title 에서 괄호 및 괄호 내 정보를 없앰
    # re.sub 함수를 사용하여 패턴에 맞는 부분을 빈 문자열로 대체
    result = re.sub(r'\(.*?\)', '', title)
    return result.rstrip()

with open("../JW_openai_credential_gpt35.txt", 'r') as f:
    api_key = f.read()
os.environ["OPENAI_API_KEY"] = api_key


vectorstore = Chroma(persist_directory="visitjeju_db_v3", embedding_function=OpenAIEmbeddings())


text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)


len_docs = len(vectorstore.get()['documents'])


# chunks 저장 - (split한 텍스트, metadata)
chunks = []
for i in tqdm(range(len_docs)):
    metadata = vectorstore.get()['metadatas'][i]['title']
    split_texts = text_splitter.split_text(vectorstore.get()['documents'][i])
    for split in split_texts:  # title clean 하여 chunk 에 삽입
        chunks.append((split, title_cleaner(metadata)))

# breakpoint()
# 쿼리 - 제목 유사도 검색 위해 "titles" collection 생성
unique_titles = list(set([c[1] for c in chunks]))

Chroma.from_texts(
    texts=unique_titles, # titles 
    embedding=OpenAIEmbeddings(),
    persist_directory="visitjeju_db_v6",
    collection_name="visitjeju_titles"
)

# chunks 를 이용해 새로운 vectorstore 생성. 
Chroma.from_texts(
    texts=[c[0] for c in chunks],
    metadatas=[{"title": c[1]} for c in chunks],
    embedding=OpenAIEmbeddings(),
    persist_directory="visitjeju_db_v6",
    collection_name="visitjeju_contexts"
)
        




