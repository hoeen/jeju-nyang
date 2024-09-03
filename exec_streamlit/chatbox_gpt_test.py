from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
import re
from typing import List
from itertools import permutations
import streamlit as st
from streamlit_chatbox import *
import time
import simplejson as json
import textwrap
from RAG.selfqueryingretriever_contents import SelfQueryingRetrieverContents
from RAG.selfqueryingretriever_title import SelfQueryingRetrieverTitle

import os

from recommendation.item_base_rec import get_recommendations_for_item




# 1. 의도분류
class Search(BaseModel):
    """추천, 기타 중 하나로 분류된 결과를 반환 -> 장소추천, 동선추천, 정보검색, 기타"""

    questionType: str = Field(
        ...,
        description="질문의 목적을 추천, 기타 중 하나로 반환",
    )
    rec_category: str = Field (None, description="장소추천, 동선추천 중 하나를 반환")
    keywords: List[str] = Field(None, description="핵심 키워드")
    place_category: str = Field (None, description="'장소를 모르는 상태에서의 질문','특정 장소에 관한 질문' 중 하나를 반환")
    etc_category: str = Field (None, description="정보검색, 기타 중 하나를 반환")
    

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def main():
    with open("../JW_openai_credential_gpt35.txt", 'r') as f:
        api_key = f.read()
    os.environ["OPENAI_API_KEY"] = api_key

    system = """
        당신은 질문의 의도 파악 분류기입니다.

        ## 지시사항 :
        - 당신은 주어지는 질문의 유형을 '추천', '기타' 중 하나로 분류한 결과를 questionType으로 반환해야 합니다.
        - 만약 questionType이 '추천'이라면, '장소추천', '동선추천' 중 하나로 분류하여 rec_category에 넣습니다. 또한 질문 속에서 등장한 장소들을 핵심 키워드로 추출하여 keywords에 넣으세요.
        - 만약 질문이 추천을 요구하지 않는 기타 질문이라면, etc_category에 '정보검색', '기타'중 하나로 분류하세요.
        - 만약 rec_category가 '장소추천' 이거나 etc_category가 '정보검색'이라면, 질문 속에 고유명사로 장소가 명시된 유무에 따라 place_category를 나눕니다. 고유명사 장소가 있는 경우에는 '특정 장소에 관한 질문', 없는 경우에는 '장소를 모르는 상태에서의 질문'으로 분류하여 넣으세요.

            --
        ## 예시 1. 
        입력받은 질문이 "제주도에서 맛있는 이탈리안 레스토랑 추천해줄 수 있나요?" 라면
        "questionType": "추천","rec_category": "장소추천", "keywords": ["제주", "이탈리안 레스토랑"],"place_category": "장소를 모르는 상태에서의 질문" 으로 저장합니다.
        


    """

    prompt = ChatPromptTemplate.from_messages([
            ("system", textwrap.dedent(system)),
            ("human", "{question}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


    structured_llm = llm.with_structured_output(Search)
    query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm

######
#2. 기본답변
    basicprompt = ChatPromptTemplate.from_messages(
        [
            ("system", '당신은 주어진 질문에 짧고 정확하게 답변하는 역할을 하는 챗봇입니다. 당신은 주어진 질문에 대한 답변만을 반환해야 합니다.'),
            ("human", "{question}"),
        ]
    )
    basicllm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    chain = {"question": RunnablePassthrough()} | basicprompt | basicllm | StrOutputParser()



######
# RAG

# load vectorstore
    vectorstore = Chroma(persist_directory="./vectorstores/visitjeju_db_place_food_shopping_stay", embedding_function=OpenAIEmbeddings())
# Retrieve and generate using the relevant snippets of the blog.
    retriever = (
        vectorstore
        .as_retriever(k=3)
        # .as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25})
    )

    def format_docs(docs):    
        return "\n\n".join(doc.page_content for doc in docs)


    template = """
        You are a Korean assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        Question: {question}
        Context: {context}
        Answer: 
    """

    ragprompt = ChatPromptTemplate.from_template(textwrap.dedent(template))

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | ragprompt
        | llm
        | StrOutputParser()
    )

######
#ver3
    system_prompt = """당신은 주어진 추천 결과를 재구성해 User에게 적합한 동선을 추천하고, 설명해주는 챗봇입니다. 
        ## 지시사항 :
        주어진 추천 리스트의 요소에 대해 위치기반으로 동선을 계획하세요. """ #정확한 위치와 정확한 최신 정보를 얻을 수 있다면 이를 함께 출력하세요 또한 네이버 지도 기준으로 리뷰가 많은 카페가 있다면 정확한 메뉴, 가격등을 함께 소개해주세요.
    recprompt = ChatPromptTemplate.from_messages(
        [
            ("system", textwrap.dedent(system_prompt)),
            ("human", "{question}"),
        ]
    )
    recllm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    rec_chain = {"question": RunnablePassthrough()} | recprompt | recllm



####
# RAG for 동선 추천

# VectorStore 및 검색기 설정
    vectorstore = Chroma(persist_directory="./vectorstores/wifilist_lat_db_v2", embedding_function=OpenAIEmbeddings(),
                         create_collection_if_not_exists=False)
    retriever = vectorstore.as_retriever(k=1)


# 문서 형식 지정
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    recragtemplate = """
        당신은 주어진 리스트를 재구성해 User에게 적합한 동선과 거리를 추천하는 챗봇입니다. wifilist_lat_db_v2 Vector Store를 잘 참고해서, apGroupName이 정확히 일치하는 내용만을 문서로 뽑으세요. Vector Store 속 정확한 latitude와 longitude를 바탕으로 답하세요. 모든 가능한 조합에서, latitude와 longitude 차이의 절대값의 합, 즉 맨하탄 거리가 가장 짧은 것부터 동선을 선택하는 알고리즘을 따라야 합니다.

### 예시 1:
    Question: 나는 제주도에서 여행중이야. 출발지는 '천지연폭포'이고, 이후 내가 방문할 곳은 '세화 해변', '협재 해변(협재해수욕장)', '제주서문시장' 이야. 제일 가까운 곳부터 가려고 하는데, 최적의 동선을 출력해줘!
    Context: {context}


    Answer:
    wifilist_lat_db_v2라는 vector store에서, 출발지 '천지연폭포'의 latitude는 33.246139이고 longitude는 126.55595이다. 
    방문할 장소는 '세화 해변', '협재 해변(협재해수욕장)', '제주서문시장' 이다.
    방문할 장소들의 latitude, longitude를 각각 wifilist_lat_db_v2에서 찾아본다. '세화 해변' (latitude: 33.525276, longitude:126.859629), "협재 해변(협재해수욕장)" (latitude: 33.39476, longitude: 126.241228), "제주서문시장" (latitude: 33.511121,longitude:126.51778)이고
    '천지연폭포'와 가장 가까운 곳을 찾기 위해 천지연 폭포와 방문할 장소들의 latitude, longitude 차이의 합을 구한다. 

    '천지연폭포'로부터 방문할 장소들까지의 latitude, longitude 차이:
    1. 세화 해변 (33.246139 - 33.525276, 126.55595 - 126.859629) -> 맨해튼 거리: -(-0.263886) + (-(-0.303679)) = 0.567565
    2. 협재 해변(협재해수욕장) (33.246139 - 33.39476, 126.55595 - 126.241228) -> 맨해튼 거리: -(-0.148621) + 0.314722 = 0.463343
    3. 제주서문시장 (33.246139 - 33.511121, 126.55595 - 126.51778) -> 맨해튼 거리: -(-0.264982) + 0.03817 = 0.303152

    따라서 처음으로 방문할 곳은 맨해튼거리가 0.303152로 가장 짧은 '제주서문시장'이다. 다음은 '제주서문시장'에서부터, 방문할 장소 리스트속 아직 방문하지 않은 원소인 '세화 해변', '협재 해변(협재해수욕장)'까지의 맨해튼 거리를 구한다.

    "제주서문시장"부터 리스트속 원소까지의 latitude, longitude 차이:
    1. 세화 해변 (33.511121 - 33.525276, 126.51778 - 126.859629) -> 맨해튼 거리: -(-0.014155) + -(-0.341849) = 0.356004
    2. 협재 해변(협재해수욕장) (33.511121- 33.39476, 126.51778 - 126.241228) -> 맨해튼 거리: 0.11635 + 0.276552 = 0.392902

    따라서 '제주서문시장' 다음으로 방문할 곳은 둘 중 맨해튼 거리가 더 짧은 '세화 해변' 이다.
    마지막으로는 남아있는 원소인 '협재 해변(협재해수욕장)'를 방문한다.




### 새로운 질문:
    Question: {question}
    Context: {context}

    Answer: 

    """


#Answer:
# 위도와 경도 기준으로 답변드리겠습니다. 출발지 '천지연폭포'에서 맨해튼 거리가 0.303152로 가장 짧은 '제주서문시장'을 먼저 방문하고, 이후 '제주서문시장'에서 거리가 0.356004로 가장 가까운 '세화 해변'을 방문한 뒤, '협재 해변(협재해수욕장)'을 방문하시는 것을 추천드립니다.

    recragprompt = ChatPromptTemplate.from_template(textwrap.dedent(recragtemplate))

    rec_rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | recragprompt
        | llm
        | StrOutputParser()
    )

# # 질문에 대한 답변 생성
# result = rec_rag_chain.invoke("한라수목원에서 출발할거야. 이후 '광치기해변', '표선 해비치 해변', '하도 해변' 어떤 순서로 가는 것이 최적일까? 동선을 추천해줘.")
# print(result)


#######
    chat_box = ChatBox()

    with st.sidebar:
        st.subheader('start to chat using streamlit')
        streaming = st.checkbox('streaming', True)
        in_expander = st.checkbox('show messages in expander', True)
        show_history = st.checkbox('show history', False)

        st.divider()

        btns = st.container()

        file = st.file_uploader(
            "chat_history.json",
            type=["json"]
        )

        # Clear history button
        if st.button("clear_history"):
            chat_box.init_session(clear=True)
            st.experimental_rerun()

        # Load history from file (optional)
        if file is not None:
            try:
                data = json.load(file)
                chat_box.from_dict(data)
            except json.JSONDecodeError:
                st.error("Error loading chat history. Please ensure it's valid JSON.")

    chat_box.init_session()
    chat_box.output_messages()

# RAG 인스턴스 
    selfqueryingretrieverTitle = SelfQueryingRetrieverTitle(llm, 'vectorstores/visitjeju_db_place_food_shopping_stay')
    selfqueryingretrieverContents = SelfQueryingRetrieverContents(llm, 'vectorstores/visitjeju_db_place_food_shopping_stay')


    if query := st.chat_input('제주 여행지 관련 질문을 해 주세요.'):
        chat_box.user_say(query)
        if streaming:
            # product_id, sim_mat_path, data_path
            llm_result = query_analyzer.invoke(query)
            print(llm_result.questionType, llm_result.rec_category, llm_result.keywords, llm_result.place_category, llm_result.etc_category)


            if llm_result.questionType == "추천":
                if llm_result.rec_category == "장소추천":
                    try:
                        print("rec")
                        recommended_items = get_recommendations_for_item(llm_result.keywords[0], 'sim_mat_240517.pkl', 'filtered_data_240517.csv')
                        generator = rec_chain.invoke(recommended_items)
                    except:
                        
                        # RAG logic으로 변경 필요
                        if llm_result.place_category == '장소를 모르는 상태에서의 질문':
                            print("rag_장소를 모르는 상태에서의 질문")
                            title = selfqueryingretrieverContents.query_analyzer.invoke({"question": query})
                            rag_chain = selfqueryingretrieverContents.create_rag_chain(title.place)
                        else:
                            print("rag_장소에대한질문")
                            place = selfqueryingretrieverTitle.query_analyzer.invoke({"question": query})
                            rag_chain = selfqueryingretrieverTitle.create_rag_chain(place.place)
                            
                        if rag_chain is not None:
                            generator = rag_chain.invoke(query)
                        else:
                            generator = "질문하신 장소에 대한 정보를 찾을 수 없습니다."
                elif llm_result.rec_category == "동선추천": 
                    generator = rec_rag_chain.invoke(query) #위치기반 동선추천
                else: 
                    generator = chain.invoke(query)
            else:
                if llm_result.etc_category == "정보검색":
                    if llm_result.place_category == '장소를 모르는 상태에서의 질문':
                        print("정보검색_rag_장소를 모르는 상태에서의 질문")
                        title = selfqueryingretrieverContents.query_analyzer.invoke({"question": query})
                        rag_chain = selfqueryingretrieverContents.create_rag_chain(title.place)
                    else:
                        print("정보검색_rag_장소에대한질문")
                        place = selfqueryingretrieverTitle.query_analyzer.invoke({"question": query})
                        rag_chain = selfqueryingretrieverTitle.create_rag_chain(place.place)

                    if rag_chain is not None:
                        generator = rag_chain.invoke(query)
                    else:
                        generator = "질문하신 장소에 대한 정보를 찾을 수 없습니다."
                else:
                    generator = chain.invoke(query)


            elements = chat_box.ai_say(
                [
                    # you can use string for Markdown output if no other parameters provided
                    Markdown("thinking", in_expander=in_expander,
                            expanded=True, title="answer"),

                ]
            )
            time.sleep(1)
            text = ""
            # for x, docs in enumerate(list(generator.content)): 
            # breakpoint()
            if hasattr(generator, 'content'):
                text = generator.content
            else:
                for docs in generator: 
                    text += docs
                    chat_box.update_msg(text, element_index=0, streaming=True)
            # update the element without focus
            chat_box.update_msg(text, element_index=0, streaming=False, state="complete")
        


    btns.download_button(
        "Export Markdown",
        "".join(chat_box.export2md()),
        file_name=f"chat_history.md",
        mime="text/markdown",
    )

    btns.download_button(
        "Export Json",
        chat_box.to_json(),
        file_name="chat_history.json",
        mime="text/json",
    )

    if btns.button("clear history"):
        chat_box.init_session(clear=True)
        st.experimental_rerun()

    if show_history:
        st.write(chat_box.history)
