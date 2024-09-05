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

import os
import pdb
from recommendation.item_base_rec import get_recommendations_for_item
from recommendation.route_rec import *
from RAG.selfqueryingretriever import SelfQueryingRetriever

# from nearest import nearest
#######

class Search(BaseModel):
    """정보 검색, 장소 추천, 동선추천, 기타 중 하나로 분류된 결과를 반환"""

    questionType: str = Field(
        ...,
        description="정보 검색, 장소 추천, 동선추천, 기타 중 하나를 반환",
    )
    keywords: List[str] = Field(None, description="핵심 키워드")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():

# load vectorstore
    title_vectorstore = Chroma(persist_directory="vectorstores/visitjeju_db_v6", collection_name="visitjeju_titles", embedding_function=OpenAIEmbeddings())
    vectorstore = Chroma(persist_directory="vectorstores/visitjeju_db_v6", collection_name="visitjeju_contexts", embedding_function=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
    retriever = (
        vectorstore
        .as_retriever(k=1)
        # .as_retriever(search_type="mmr", search_kwargs={'k': 6, 'lambda_mult': 0.25})
    )

# 1. 의도분류
    system = """
        당신은 제주도의 여행 가이드입니다.
        당신은 주어지는 질문에서 '정보 검색', '장소 추천', '동선추천', '기타' 중 하나로 분류된 결과를 반환해야 합니다.
        그 결과를 주어지는 쿼리로 반환하십시오. 만약 처리하기 애매한 것이 있다면, '기타'로 분류하십시오.
        주어지는 질문이 '정보 검색'이나 '장소 추천', '동선추천' 이라면, 질문 속 등장한 장소들을 핵심 키워드로 추출하십시오.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
    structured_llm = llm.with_structured_output(Search)
    query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm


######
#2. RAG - RAG.selfqueryingretriever 의 class import 로 대
    # templacte = """
    #     You are a Korean assistant for question-answering tasks. 
    #     Use the following pieces of retrieved context to answer the question. 
    #     If you don't know the answer, just say that you don't know. 
    #     Use three sentences maximum and keep the answer concise.
    #     Question: {question}
    #     Context: {context}
    #     Answer: 
    # """
    # ragprompt = ChatPromptTemplate.from_template(template)
    # rag_chain = (
    #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #     | ragprompt
    #     | llm
    #     | StrOutputParser()
    # )

######
#ver3
    recprompt = ChatPromptTemplate.from_messages(
        [
            ("system", '당신은 주어진 추천 결과를 재구성해 User에게 적합한 동선을 추천하고, 설명해주는 챗봇입니다. 주어진 추천 리스트의 요소에 대해 정확한 위치와 정확한 최신 정보를 얻을 수 있다면 이를 함께 출력하고, 위치기반으로 동선을 계획하세요. 또한 네이버 지도 기준으로 리뷰가 많은 카페가 있다면 정확한 메뉴, 가격등을 함께 소개해주세요.'),
            ("human", "{question}"),
        ]
    )
    recllm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    rec_chain = {"question": RunnablePassthrough()} | recprompt | recllm



####
# RAG for 동선 추천
    vectorstore = Chroma(persist_directory="wifilist_lat_db_v2", embedding_function=OpenAIEmbeddings())
    route_retriever = vectorstore.as_retriever(k=1)

    recragtemplate = """
        당신은 주어진 리스트를 재구성해 User에게 적합한 동선을 추천하는 챗봇입니다. wifilist_lat_db_v2 Vector Store를 잘 참고해서, apGroupName이 정확히 일치하는 내용만을 문서로 뽑으세요. 정확한 위도와 경도를 바탕으로 답하세요. 
        모든 가능한 조합에서, 위도와 경도 차이의 합, 즉 총 맨하탄 거리가 가장 짧은 조합으로 동선을 선택하는 알고리즘을 따라야 합니다. 이 내용을 바탕으로 최적의 동선을 계획해 출력하시오.
        Question: {question}
        Context: {context}
        Answer: 
    """
    recragprompt = ChatPromptTemplate.from_template(recragtemplate)
    rec_rag_chain = lambda query: rec_rag_chain_logic(route_retriever, query)


# 질문에 대해 호출
    # response = rec_rag_chain("나는 한라수목원에서 출발할거야. 용담해안 카페촌거리, 표선청소년문화의집 어떤 순서로 가는 것이 최적일까? 동선을 추천해줘.")
    # print(response)


# streamlit 실행 
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
    selfqueryingretriever = SelfQueryingRetriever(llm, 'vectorstores/visitjeju_db_v6')

    if query := st.chat_input('제주 여행지 관련 질문을 해 주세요.'):
        chat_box.user_say(query)
        if streaming:
            # product_id, sim_mat_path, data_path
            llm_result = query_analyzer.invoke(query)
            print(llm_result.questionType, llm_result.keywords)

            # logic 구현
            if llm_result.questionType == "정보 검색":
                # RAG logic으로 변경 필요
                place_info = selfqueryingretriever.query_analyzer.invoke({"question": query})
                rag_chain = selfqueryingretriever.create_rag_chain(place_info.place)
                if rag_chain is not None:
                    generator = rag_chain.invoke(query)
                else:
                    generator = "질문하신 장소에 대한 정보를 찾을 수 없습니다."
            elif llm_result.questionType == "장소 추천":
                # 추천 알고리즘 돌리는 logic
                recommended_items = get_recommendations_for_item(llm_result.keywords[0], 'sim_mat_240517.pkl', 'filtered_data_240517.csv')
                print(recommended_items)
                generator = rec_chain.invoke(recommended_items)
            elif llm_result.questionType == "동선 추천":
                generator = rec_rag_chain.invoke(query)
            else:
                generator = rag_chain.invoke(query)
            # print(generator)

            elements = chat_box.ai_say(
                [
                    # you can use string for Markdown output if no other parameters provided
                    Markdown("thinking", in_expander=in_expander,
                            expanded=True, title="answer"),
                    Markdown("", in_expander=in_expander, title="references"),
                ]
            )
            time.sleep(1)
            text = ""
            if llm_result.questionType == "장소 추천":
                for x, docs in enumerate(list(generator.content)):
                    text += docs
                    chat_box.update_msg(text, element_index=0, streaming=True)
            # update the element without focus
                chat_box.update_msg(text, element_index=0, streaming=False, state="complete")
                chat_box.update_msg("\n\n".join(docs), element_index=1, streaming=False, state="complete")
    
            elif llm_result.questionType == "정보 검색":
                chat_box.update_msg(generator, element_index=0, streaming=False, state="complete")



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
