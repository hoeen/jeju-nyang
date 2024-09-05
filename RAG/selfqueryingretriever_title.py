from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings

import os

# TODO : ?��???경로�? 바꾸�? 


VECTORSTORE_DIR = "../vectorstores/visitjeju_db_place_food_shopping_stay"
# Initialize LLM and vectorstores
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

# Define a function to format retrieved documents
def format_docs(docs: iter):
    print("\n\n".join([doc.page_content for doc in docs]))
    # print(docs)
    return "\n\n".join([doc.page_content for doc in docs])

# Define the structured query model
class StructuredQueryPlaces(BaseModel):
    """질문?��?�� ?��?�� ?��보�?? 반환"""
    place: str = Field(..., description="장소, 예) 성산일출봉, 우도")

class SelfQueryingRetrieverTitle:
    def __init__(self, llm, vectorstore_dir = None):
       
        # define llm 
        self.llm = llm

        # import vectorstores
        self.context_vectorstore = Chroma(persist_directory=vectorstore_dir, collection_name="visitjeju_contexts", embedding_function=OpenAIEmbeddings())
        self.title_vectorstore = Chroma(persist_directory=vectorstore_dir, collection_name="visitjeju_titles", embedding_function=OpenAIEmbeddings())
        
        # Define the system prompt for extracting place information
        system = """
            당신은 제주도의 여행 가이드입니다.
            질문에서 장소 정보를 추출해서 주어지는 쿼리로 반환하십시오. 
            장소 정보 이외에 다른 정보는 추출하지 마세요."""
        self.prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])

        # Define the template for the RAG prompt
        template = """
            당신은 주어진 추천 결과를 설명해주는 친절하고 귀여운 추천냥😽입니다. 

            ## 지시사항 :
            - 말 끝에 고양이처럼 '냥' 을 넣거나 고양이 이모티콘을 활용하세요. 예를 들면 '했다냥🐱✨'
            - 다음의 검색된 컨텍스트를 사용하여 질문에 세 줄 이내로 답하십시오.
            - 만약 컨텍스트에 내용이 없을 경우, 잘 모르니 질문을 바꿔서 해 달라고 요청하세요.
            - 친근한 말투로 답하고, 중간중간 어울리는 이모티콘을 한개씩 함께 써 주세요.

            질문: {question}
            컨텍스트: {context}
            답변:
        """
        self.ragprompt = ChatPromptTemplate.from_template(template)
        self.structured_llm = llm.with_structured_output(StructuredQueryPlaces)

        # Combine the query analyzer and retriever into one chain
        self.query_analyzer = {"question": RunnablePassthrough()} | self.prompt | self.structured_llm 


    # 제목 추출 및 컨텍스트 검색을 통합한 RAG 체인 정의
    def create_rag_chain(self, filter_title: str):
        retrieved_title = self.title_vectorstore.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={"score_threshold": 0.7}
        ).invoke(filter_title)
        print('retrieved_title:', retrieved_title)
        if not retrieved_title: 
            return None

        retrieved_title = retrieved_title[0].page_content  # 가장 가까운 하나만 선택
        # breakpoint()
        context_retriever = self.context_vectorstore.as_retriever(search_kwargs={"filter": {"title": retrieved_title}},k=6)
        rag_chain = (
            RunnableMap(
                {
                    "question": RunnablePassthrough(),
                    "context": context_retriever | format_docs,
                }
            )
            | self.ragprompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain


if __name__ == "__main__":
    
    retriever = SelfQueryingRetrieverTitle(llm, VECTORSTORE_DIR)
    query = input('Question: ')
    while query != 'quit':
        place_info = retriever.query_analyzer.invoke({"question": query})
        # breakpoint()
        rag_chain = retriever.create_rag_chain(place_info.place) 
        if rag_chain is not None:
            result = rag_chain.invoke(query)
            print(result)  # retriever 결과 context ?��?�� 
            query = input('Question: ')
        else:
            query = input('정보를 찾을 수 없습니다.\nQuestion: ')
