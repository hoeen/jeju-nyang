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
    

    category: str = Field(..., )

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata









