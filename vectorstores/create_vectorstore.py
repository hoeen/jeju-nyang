import os, time 
from tqdm import tqdm

from selenium import webdriver

from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLSectionSplitter
from langchain_openai import ChatOpenAI
from langchain_community.document_transformers.html2text import Html2TextTransformer

from bs4 import BeautifulSoup


URLs = []
with open('../visitjeju_places_url_list.txt', 'r') as f:
    for line in f.readlines():
        URLs.append(line.strip())
        # break




# loader = WebBaseLoader(
#     web_paths=URLs[:10],
#     bs_kwargs=dict(
#         parse_only=SoupStrainer(
#             class_=("tag_area", 
#                     "add2020_detail_box_in", 
#                     "basic_information",
#                     "add2020_detail_side_info")
#         )
#     ),
#     requests_per_second=100
# )


info_list = []
title_list = []

# chrome 창 열기
driver = webdriver.Chrome()  
 

for url in tqdm(URLs[:2]):
    # 사이트에서 특정 class 내 html 정보만 추출하기
    try:
        driver.get(url)
        time.sleep(1)
        # URL에서 HTML 가져오기
        html = driver.page_source 

        # selenium 으로 접근한 HTML bs4로 파싱
        soup = BeautifulSoup(html, "html.parser")
        info = ( 
            soup.find(class_='tag_area').get_text(' ', strip=True) + '\n' +
            soup.find(class_='add2020_detail_box_in').get_text(' ', strip=True) + '\n' +
            soup.find(class_='basic_information').get_text(' ', strip=True) + '\n' +
            soup.find(class_='add2020_detail_side_info').get_text(' ', strip=True)
        )
        title = soup.find(class_='sub_info_title').get_text(' ', strip=True)
        info_list.append(info)
        title_list.append(title)
    except:
        print('Error occured while parsing data from url: %s' % url)


# TODO: 적절한 splitter 로 나눈다음, 관광지 태깅이 되도록 하기


Chroma.from_texts(
    texts=info_list,
    metadatas=[{"title": t} for t in title_list],
    embedding=OpenAIEmbeddings(),
    persist_directory="visitjeju_db_v3"
)

# breakpoint()


# # Chroma 벡터를 로드할 때 메타데이터를 활용합니다.
# chroma = Chroma(embedding_function=OpenAIEmbeddings())

# chroma.from_documents(


# for url, (title, doc) in tqdm(url_document_mapping.items()):
#     # 청크로 나누기
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
#     splits = text_splitter.split_text(doc)
    
#     # 각 청크를 하위 구조에 추가
#     for split in splits:
#         substructure.add_text(split)

#     # 벡터 적재
#     chroma.load_vector(title, substructure)

# # 벡터 저장
# vectorstore = chroma.persist(persist_directory="visitjeju_test")

# # docs[0].page_content.split()[0]

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
# splits = text_splitter.split_documents(docs)

# save vectorstore
# vectorstore = Chroma.from_documents(
#     documents=splits, 
#     embedding=OpenAIEmbeddings(), 
#     # persist_directory="visitjeju_db"
#     persist_directory="visitjeju_db_v2"
# )

