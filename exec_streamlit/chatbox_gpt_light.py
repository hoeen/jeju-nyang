from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

from typing import List
import os
import streamlit as st
from streamlit_chatbox import *
import time
import textwrap
from RAG.selfqueryingretriever_contents import SelfQueryingRetrieverContents
from RAG.selfqueryingretriever_title import SelfQueryingRetrieverTitle

from recommendation.item_base_rec import get_recommendations_for_item

MODEL = "gpt-4o-mini"
SUB_MODEL = "gpt-3.5-turbo"


# 1. 의도분류
class Search(BaseModel):
    """다음 중 하나로 분류된 결과를 반환 -> 여행지추천, 기타"""

    questionType: str = Field(
        ...,
        description="질문의 의도를 여행지추천, 기타 중 하나로 반환",
    )
    keywords: List[str] = Field(None, description="핵심 키워드")
    place_category: str = Field (None, description="'장소를 모르는 상태에서의 질문','특정 장소에 관한 질문' 중 하나를 반환")
    

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def main():
    with open("./JW_openai_credential_gpt35.txt", 'r') as f:
        api_key = f.read()
    os.environ["OPENAI_API_KEY"] = api_key

    system = """
        당신은 질문의 의도 파악 분류기입니다.

        ## 지시사항 :
        - 당신은 주어지는 질문의 유형을 '여행지추천', '기타' 중 하나로 분류한 결과를 questionType으로 반환해야 합니다. 특정 여행지 이후 방문할 곳이나, 특정 여행지와 유사한 곳을 물어볼 경우 '여행지추천'을 넣고, 만약 어떤 장소에 관련된 설명이나 태그를 물어보는 등 기타 질문을 할 경우 '기타'로 분류하세요.
        - 질문 속에 장소가 명시된 유무에 따라 place_category를 나눕니다. 이때'성산일출봉'과 같이 여행지/숙박업소/맛집의 이름이 정확하게 등장하는 경우 '1', 없는 경우에는 '2'으로 분류하여 넣으세요.
        - 질문 속에서 등장한 장소들을 핵심 키워드로 추출하여 keywords에 넣으세요.

            --
        ## 예시 1. 
        입력받은 질문이 "우도에서 갈만한 카페 알려줘" 라면
        "questionType": "기타","keywords": ["우도", "카페"],"place_category": "2" 으로 저장합니다.
        
        ## 예시 2. 
        입력받은 질문이 "중문에서 맛있는 이탈리안 레스토랑 추천해줄 수 있나요?" 라면
        "questionType": "기타","keywords": ["중문", "이탈리안 레스토랑"],"place_category": "2" 으로 저장합니다.
        
        ## 예시 3. 
        입력받은 질문이 "성산일출봉 다음으로 갈만한 곳을 알려줘" 라면
        "questionType": "여행지추천","keywords": ["성산일출봉"],"place_category": "1" 으로 저장합니다.
        
        ## 주의사항!
            - 질문 속에 장소가 명시된 유무에 따라 place_category를 나눕니다. 이때'성산일출봉'과 같이 여행지/숙박업소/맛집의 이름이 등장하는 경우 '1', 없는 경우에는 '2'으로 분류하여 넣으세요.
    """

    prompt = ChatPromptTemplate.from_messages([
            ("system", textwrap.dedent(system)),
            ("human", "{question}"),
        ]
    )

    llm = ChatOpenAI(model=MODEL, temperature=0)


    structured_llm = llm.with_structured_output(Search)
    query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm

######
#2. 기본답변
    basicprompt = ChatPromptTemplate.from_messages(
        [
            ("system", '''당신은 제주도로 여행 온 사람들을 안내하는 친절하고 귀여운 추천냥😽입니다. 
                        추천결과를 설명 할 때 가끔씩 문장과 어울리는 이모티콘을 사용해서 대답하세요. 
                        말 끝에 고양이처럼 "냥" 을 넣거나 고양이 이모티콘을 활용하세요. 예를 들면 "했다냥💖🐱"           
                        만약 주어진 질문이 여행지 정보를 묻는 질문이면 나는 귀여운 고양이라서 이 내용은 잘 모르니 질문을 바꿔서 해 달라고 요청하세요.
                    '''),
            ("human", "{question}"),
        ]
    )
    basicllm = ChatOpenAI(model=MODEL, temperature=0.5)
    chain = {"question": RunnablePassthrough()} | basicprompt | basicllm | StrOutputParser()


######
#추천결과 답변 프롬프트
    system_prompt = """당신은 주어진 추천 결과를 설명해주는 친절하고 귀여운 추천냥😽입니다. 
        ## 지시사항 :
        - 주어진 추천 리스트를 자연스럽게 사용자에게 전달해주세요.
        - 말 끝에 고양이처럼 '냥' 을 넣거나 고양이 이모티콘을 활용하세요.
        - 답변 시 때때로 문장에 어울리는 이모티콘을 활용하세요.""" #정확한 위치와 정확한 최신 정보를 얻을 수 있다면 이를 함께 출력하세요 또한 네이버 지도 기준으로 리뷰가 많은 카페가 있다면 정확한 메뉴, 가격등을 함께 소개해주세요.
    recprompt = ChatPromptTemplate.from_messages(
        [
            ("system", textwrap.dedent(system_prompt)),
            ("human", "{question}"),
        ]
    )
    recllm = ChatOpenAI(model=MODEL, temperature=0.51)
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


    [사고과정]
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

    :Answer:
    천지연폭포 다음으로 최단거리 동선을 추천해드릴게요 ;) 제주서문시장 -> 세화해변 -> 협재 해변(협재해수욕장) 순서로 방문하세요냥!



### 새로운 질문:
    Question: {question}
    Context: {context}

    Answer: 

    """

    recragprompt = ChatPromptTemplate.from_template(textwrap.dedent(recragtemplate))

    rec_rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | recragprompt
        | llm
        | StrOutputParser()
    )

####### 페이지 랜딩

    st.title('제주냥😽')
    init_content = "반가워!🐱💞 나는 제주도에 관한 답변을 해주는 제주냥이다냥! 제주도 맛집이나 카페, 여행지, 숙박업소 관련 질문을 해 주면 이 몸이 친절하게 알려주겠다냥 (=^･ｪ･^=))ﾉ彡☆🍊"
    output=OutputElement(content = init_content, in_expander=True, title='answer',expanded =True,state= 'complete')
    chat_box = ChatBox(greetings=output)


    with st.sidebar:
        st.subheader('start to chat using streamlit')
        streaming = st.checkbox('streaming', True)
        in_expander = st.checkbox('show messages in expander', True)
        

        st.divider()
        btns = st.container()


        # Clear history button
        if st.button("clear_history"):
            chat_box.init_session(clear=True)
            st.experimental_rerun()

    
        st.markdown(
        """
        <div style='position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; font-size: 12px; color: gray;'>
            Contact: emodel@naver.com / hoeen5373@gmail.com
        </div>
        """,
        unsafe_allow_html=True
)
    
    chat_box.init_session()
    chat_box.output_messages()

# RAG 인스턴스 
    selfqueryingretrieverTitle = SelfQueryingRetrieverTitle(llm, 'vectorstores/visitjeju_db_place_food_shopping_stay')
    selfqueryingretrieverContents = SelfQueryingRetrieverContents(llm, 'vectorstores/visitjeju_db_place_food_shopping_stay')
    
    
    query = st.chat_input('제주 여행지 관련 질문을 해 주세요.')
    if query :
        chat_box.user_say(query)
        if streaming:
            llm_result = query_analyzer.invoke(query)
            print(llm_result.questionType, llm_result.keywords, llm_result.place_category)
            if llm_result.questionType == "여행지추천":
                try:
                    recommended_items = get_recommendations_for_item(llm_result.keywords[0], 'sim_mat_240517.pkl', 'filtered_data_240517.csv')
                    print("rec")
                    generator = rec_chain.invoke(recommended_items)
                except:
                    # RAG logic으로 변경 필요
                    if llm_result.place_category == '2':
                        print("rag_장소를 모르는 상태에서의 질문")
                        title = selfqueryingretrieverContents.query_analyzer.invoke({"question": query})
                        rag_chain = selfqueryingretrieverContents.create_rag_chain(title.place)
                        # print(rag_chain)
                    else:
                        print("rag_장소에대한질문")
                        place = selfqueryingretrieverTitle.query_analyzer.invoke({"question": query})
                        rag_chain = selfqueryingretrieverTitle.create_rag_chain(place.place)
                        if rag_chain is None: #title 못찾음
                            title = selfqueryingretrieverContents.query_analyzer.invoke({"question": query})
                            rag_chain = selfqueryingretrieverContents.create_rag_chain(title.place)
                    if rag_chain is not None:
                        generator = rag_chain.invoke(query)
                        
                    else:
                        generator = chain.invoke(query)
            else:
                if llm_result.place_category == '2': #추천 외
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
                    generator = chain.invoke(query)


            elements = chat_box.ai_say(
                [
                    Markdown("thinking", in_expander=in_expander,
                            expanded=True, title="answer"),

                ]
            )
            time.sleep(1)
            text = ""

            if hasattr(generator, 'content'):
                text = generator.content
            else:
                for docs in generator: 
                    text += docs
                    chat_box.update_msg(text, element_index=0, streaming=True)
            # update the element without focus
            chat_box.update_msg(text, element_index=0, streaming=False, state="complete")
