import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import configparser
import streamlit as st


# 저장된 유사도 행렬을 로드하는 함수
def load_similarity_matrix(sim_mat_path):
    
    with open('models/similarity_matrix/{}'.format(sim_mat_path), 'rb') as f:
        similarity_matrix = pickle.load(f)
    return similarity_matrix

# 저장된 데이터를 로드하는 함수
def load_agg_data(filtered_data_path):
    agg_data=pd.read_csv('models/data/{}'.format(filtered_data_path))
    return agg_data


# 주어진 아이템에 대해 유사한 다른 아이템을 추천하는 함수
def get_recommendations_for_item(product_id, sim_mat_path, filtered_data_path):
    similarity_matrix = load_similarity_matrix(sim_mat_path)
    data = load_agg_data(filtered_data_path)
    # 유사도 행렬에서 주어진 아이템에 대한 유사도 정보 가져오기

    # 'movingPath' 컬럼의 각 항목을 개별 아이템으로 분리
    data['movingPath'] = data['movingPath'].apply(eval)
    data = data.explode('movingPath')

    # print(data['product.1'].unique().tolist(), product_id)
    item_list = data['movingPath'].unique().tolist() ###explode빼먹음?

    
#     if product_id in item_list: #실제 아이템 ID(현재는 product_id로 작성)를 받았을 때, 내부 인덱스 찾아가기
    item_similarities = similarity_matrix[item_list.index(product_id)]
        
        # 유사도가 가장 높은 상위 5개의 아이템 추출
    similar_item_indices = np.argsort(item_similarities)[-6:-1][::-1]  # 자기 자신은 제외
    recommended_items = [item_list[i] for i in similar_item_indices]
#     else:
#         recommended_items = 'out of value'
    
    # 유사도가 높은 아이템의 인덱스를 반환
    return recommended_items

