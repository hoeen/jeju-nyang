import matplotlib.pyplot as plt

from konlpy.tag import Mecab
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

def korean_bleu_score(reference, candidate):
    # Mecab 형태소 분석기 초기화
    mecab = Mecab()
    
    # 형태소 단위로 토큰화
    ref_tokens = mecab.morphs(reference)
    cand_tokens = mecab.morphs(candidate)
    
    # BLEU 점수 계산 (1-gram부터 4-gram까지)
    weights_list = [
        (1, 0, 0, 0),   # 1-gram
        (0.5, 0.5, 0, 0),  # 1, 2-gram
        (0.33, 0.33, 0.33, 0),  # 1, 2, 3-gram
        (0.25, 0.25, 0.25, 0.25)  # 1, 2, 3, 4-gram
    ]
    
    bleu_scores = []
    for weights in weights_list:
        score = sentence_bleu([ref_tokens], cand_tokens, weights=weights)
        bleu_scores.append(score)
    
    return {
        '1-gram BLEU': bleu_scores[0],
        '1-2gram BLEU': bleu_scores[1],
        '1-3gram BLEU': bleu_scores[2],
        '1-4gram BLEU': bleu_scores[3]
    }

# 데이터셋
datasets = [
    {
        "question": "성산일출봉이 유네스코에 등재된 이유는?",
        "original_answer": "성산일출봉은 빼어난 경관과 지질학적 가치를 인정받아 2007년 7월 2일 유네스코 세계자연유산에 등재되었습니다.",
        "llm_generated_answer": "성산일출봉은 독특한 화산 구조와 생태계, 그리고 아름다운 자연경관으로 인해 유네스코 세계자연유산에 등재되었습니다.",
        "chatbot_response": "성산일출봉은 빼어난 경관과 지질학적 가치를 인정받아 유네스코 세계자연유산으로 등재되었답니다냥🐾✨"
    },
    {
        "question": "성산일출봉의 명칭 유래는?",
        "original_answer": "'성산'은 거대한 성과 같은 모습에서 유래하였고, '일출봉'은 정상에서 바라보는 일출이 장관이기 때문에 붙여진 이름입니다.",
        "llm_generated_answer": "성산일출봉의 명칭은 일출이 아름답게 보이는 성산에서 유래하였으며, '성산'은 성스러운 산이라는 뜻을 가지고 있습니다.",
        "chatbot_response": "성산일출봉은 해가 뜨는 모습이 장관이라 하여 '일출봉(日出峰)'이라는 이름이 붙여졌고, 그 모습이 거대한 성과 같다고 해서 '성산(城山)'이라는 이름도 함께 가지고 있답니다냥🐾✨"
    },
    {
        "question": "성산일출봉 정상에서 무엇을 볼 수 있나요?",
        "original_answer": "정상에서는 너비가 약 8만 평에 이르는 분화구와 제주 바다의 웅장한 풍경을 볼 수 있습니다.",
        "llm_generated_answer": "성산일출봉 정상에서는 아름다운 동해의 일출과 주변의 푸른 바다, 그리고 제주도의 경치가 한눈에 내려다보이는 장관을 감상할 수 있습니다.",
        "chatbot_response": "성산일출봉 정상에서는 넓은 분화구와 그 뒤로 펼쳐지는 웅장한 바다의 풍경을 감상할 수 있다냥🐾✨!"
    },
    {
        "question": "성산일출봉의 등반 난이도는?",
        "original_answer": "경사도가 가파르지만 약 20분 정도면 정상에 도달할 수 있으며, 전체 탐방 시간은 1시간 미만입니다.",
        "llm_generated_answer": "성산일출봉의 등반 난이도는 중급으로, 약 30분에서 1시간 정도 소요되며, 경치가 아름다워 많은 관광객이 찾습니다.",
        "chatbot_response": "성산일출봉은 가파른 계단 길이 있지만, 넉넉히 20분이면 정상에 도착할 수 있어요냥🐾✨!"
    },
    {
        "question": "성산일출봉에 관련된 역사적 사건은?",
        "original_answer": "4·3항쟁 당시 많은 민간인이 이 지역에서 목숨을 잃었으며, 1943년에는 일본군이 해안 절벽에 요새를 구축하기도 했습니다.",
        "llm_generated_answer": "성산일출봉은 2011년 유네스코 세계자연유산으로 등재되었으며, 그 독특한 지형과 생태계는 제주도의 자연과 문화유산을 대표하는 중요한 장소입니다.",
        "chatbot_response": "성산일출봉은 1943년 일본군이 요새화하기 위해 해안절벽에 24개의 굴을 팠던 아픈 역사를 가지고 있다냥😿✨"
    },
    {
        "question": "사려니숲길의 주요 특징은 무엇인가요?",
        "original_answer": "빽빽한 삼나무와 졸참나무, 서어나무 등이 어우러진 숲길로, 청정한 공기를 마시며 트레킹을 즐길 수 있는 장소입니다.",
        "llm_generated_answer": "사려니숲길은 울창한 편백나무 숲과 맑은 공기를 즐길 수 있는 평탄한 길로, 자연 속에서 힐링할 수 있는 최적의 장소입니다.",
        "chatbot_response": "사려니숲길은 유네스코 생물권 보전지역으로, 다양한 수종과 함께 상쾌한 삼나무 향을 느끼며 트래킹을 즐길 수 있는 신성한 숲이란다냥🐾✨"
    },
    {
        "question": "사려니숲길에서 탐방 제한 구역은?",
        "original_answer": "'물찻오름'은 자연휴식년제로 탐방이 제한되었으나, 한시적으로 개방될 때도 있습니다.",
        "llm_generated_answer": "사려니숲길에서 탐방 제한 구역은 '사려니숲길 탐방로'의 일부 구간으로, 해당 구역은 자연 보호를 위해 출입이 금지되어 있습니다.",
        "chatbot_response": "사려니숲길에서 탐방 제한 구역은 '물찻오름'이니, 이곳은 탐방이 제한되니 주의해야 한다냥🐾✨!"
    },
    {
        "question": "사려니숲길의 소요 시간은?",
        "original_answer": "기본 탐방은 약 2시간 정도 걸리며, 물찻오름까지 포함하면 3시간 이상 소요됩니다.",
        "llm_generated_answer": "사려니숲길을 걷는 데 보통 2~3시간 정도 소요됩니다.",
        "chatbot_response": "사려니숲길을 탐방하려면 대중교통 이용 시 약 2시간 정도 소요된답니다냥🐾✨ 자가용으로 가면 3시간에서 3시간 30분 정도 걸린다고 해요! 즐거운 여행 되길 바랄게요냥~😊"
    },
    {
        "question": "사려니숲길의 편의시설은?",
        "original_answer": "화장실, 주차장, 안내시설 등이 있으며, 음식물 반입은 금지되지만 플라스틱 물병은 허용됩니다.",
        "llm_generated_answer": "사려니숲길에는 화장실, 주차장, 그리고 휴식 공간이 마련되어 있어 편리하게 이용할 수 있습니다.",
        "chatbot_response": "사려니숲길에는 주차 공간이 없어서 남조로변에 주차하고 탐방해야 한다냥🐾! 음식물 반입은 금지지만 플라스틱 물병은 가능하니 참고해줘냥✨."
    },
    {
        "question": "사려니숲길의 생태적 가치는?",
        "original_answer": "사려니숲길은 유네스코가 지정한 생물권 보전지역으로, 다양한 조류와 파충류의 서식지로도 유명합니다.",
        "llm_generated_answer": "사려니숲길은 다양한 식물과 동물이 서식하는 생태계의 보고로, 자연 생태계 보전과 생물 다양성 유지에 중요한 역할을 합니다.",
        "chatbot_response": "사려니숲길은 다양한 생물의 보금자리로, 유네스코 생물권 보전지역으로 지정된 만큼 생태적 가치가 아주 높다냥🐾✨"
    },
    {
        "question": "카멜리아힐의 주소는 어디인가요?",
        "original_answer": "제주 서귀포시 안덕면 병악로 166입니다.",
        "llm_generated_answer": "카멜리아힐의 주소는 제주특별자치도 서귀포시 안덕면 병악로 166입니다.",
        "chatbot_response": "카멜리아힐의 주소는 제주 서귀포시 안덕면 병악로 166이냥🐾✨! 아름다운 동백꽃과 포토 스팟이 가득하니 꼭 가보세요!"
    },
    {
        "question": "카멜리아힐의 주요 관람 시기는 언제인가요?",
        "original_answer": "연중 무휴로 운영되며, 하절기(6-8월)에는 08:30-19:00, 간절기(3-5월, 9-10월)에는 08:30-18:30, 동절기(1-2월, 11-12월)에는 08:30~18:00까지 운영됩니다.",
        "llm_generated_answer": "카멜리아힐의 주요 관람 시기는 주로 2월부터 4월까지로, 이 시기에 동백꽃이 만개하여 아름다운 경관을 제공합니다.",
        "chatbot_response": "카멜리아힐의 주요 관람 시기는 하절기(68월)와 간절기(35월, 9~10월)로, 특히 동백꽃이 피어나는 시기에 방문하면 인생샷을 남기기 좋다냥🐾✨!"
    },
    {
        "question": "카멜리아힐의 입장료는 얼마인가요?",
        "original_answer": "개인 기준 성인은 10,000원, 청소년/경로/군인은 8,000원, 어린이/장애인/보훈대상 및 4.3 유족은 7,000원입니다. 단체 요금은 성인 8,000원, 청소년/경로/군인은 7,000원, 어린이/장애인/보훈대상 및 4.3 유족은 6,000원입니다.",
        "llm_generated_answer": "카멜리아힐의 입장료는 성인 기준 약 9,000원이지만, 어린이와 노인은 할인된 요금이 적용됩니다.",
        "chatbot_response": "카멜리아힐의 입장료는 성인 10,000원, 청소년과 경로, 군인은 8,000원, 어린이와 장애인, 보훈대상 및 4.3 유족은 7,000원이래냥🐾✨!"
    },
    {
        "question": "카멜리아힐에서 유명한 포토 스팟은 무엇인가요?",
        "original_answer": "\"새소리 바람소리길\", 전통 올레길, 노란 전구가 빛나는 감성 숲길 등이 유명한 포토 스팟입니다.",
        "llm_generated_answer": "카멜리아힐에서 유명한 포토 스팟은 아름다운 동백꽃이 만개하는 시기에 볼 수 있는 '동백꽃 터널'과 '하트 모양의 벤치'입니다.",
        "chatbot_response": "카멜리아힐에서 유명한 포토 스팟은 이니스프리 CF의 배경으로 알려진 새소리 바람소리길과 노란 전구들이 반짝이는 감성적인 숲길이야냥🐾✨!"
    },
    {
        "question": "협재해수욕장에서 반려동물 동반이 가능한가요?",
        "original_answer": "네, 반려동물 동반이 가능하지만, 해수욕장 개장 기간 내에는 지정 유영 구역 입장은 통제됩니다. 현장 상황에 따라 반려동물 입장이 제한될 수도 있습니다. 목줄(2m 이내)과 배변 봉투는 필수입니다.",
        "llm_generated_answer": "협재해수욕장은 반려동물 동반이 가능하지만, 반드시 목줄을 착용하고 쓰레기는 반드시 수거해야 합니다.",
        "chatbot_response": "협재해수욕장에서 반려동물 동반은 가능하지만, 해수욕장 개장 기간 동안 유영 구역의 입욕은 통제된다는 점을 참고해야 해냥🐾! 목줄과 배변봉투는 필수니까 잊지 말고 챙기세요🐱✨!"
    },
    {
        "question": "협재해수욕장의 주요 특징은 무엇인가요?",
        "original_answer": "투명하고 에메랄드빛 바다, 은모래 백사장, 완만한 경사와 얕은 수심이 특징이며, 비양도와 석양 풍경도 인기 있습니다. 가족 단위 여행객에게 적합합니다.",
        "llm_generated_answer": "협재해수욕장은 맑고 푸른 바다와 고운 백사장이 특징이며, 수영과 해양 스포츠를 즐기기에 적합한 장소입니다.",
        "chatbot_response": "협재해수욕장은 경사가 완만하고 수심이 얕아 어린이와 가족 단위 여행객에게 딱 좋은 곳이랍니다냥🐾✨ 주변에는 다양한 카페와 맛집도 많아서 즐길 거리가 가득해요! 아름다운 바다와 은모래 백사장이 힐링을 선사해준답니다냥🐱💖"
    },
    {
        "question": "협재해수욕장 인근의 추천 관광지는 어디인가요?",
        "original_answer": "한림공원, 금능해수욕장, 금능석물원, 월령선인장자생지 등이 인근에 있습니다.",
        "llm_generated_answer": "협재해수욕장 인근의 추천 관광지는 한림공원과 협재 자연휴양림입니다.",
        "chatbot_response": "협재해수욕장 인근에는 한림공원, 금능해수욕장, 금능석물원, 월령선인장자생지 같은 멋진 관광지가 많다냥🐾✨!"
    }
]

# BLEU 점수 계산 및 평균 계산
llm_scores = {
    '1-gram BLEU': [],
    '1-2gram BLEU': [],
    '1-3gram BLEU': [],
    '1-4gram BLEU': []
}

chatbot_scores = {
    '1-gram BLEU': [],
    '1-2gram BLEU': [],
    '1-3gram BLEU': [],
    '1-4gram BLEU': []
}

for data in datasets:
    llm_bleu = korean_bleu_score(data['original_answer'], data['llm_generated_answer'])
    chatbot_bleu = korean_bleu_score(data['original_answer'], data['chatbot_response'])
    
    for key in llm_scores.keys():
        llm_scores[key].append(llm_bleu[key])
        chatbot_scores[key].append(chatbot_bleu[key])

# 평균 계산
print("LLM Generated Answer 평균 BLEU 점수:")
for key in llm_scores.keys():
    print(f"{key}: {np.mean(llm_scores[key]):.4f}")

print("제주냥 Generated Answer 평균 BLEU 점수:")
for key in chatbot_scores.keys():
    print(f"{key}: {np.mean(chatbot_scores[key]):.4f}")

# fig = plt.figure()
chatbot_score_list = np.array([np.mean(chatbot_scores[k]) for k in chatbot_scores.keys()])
llm_score_list = np.array([np.mean(llm_scores[k]) for k in llm_scores.keys()])

plt.bar(chatbot_scores.keys(), (chatbot_score_list - llm_score_list) / llm_score_list * 100)
plt.ylabel('Relative BLEU Improvement (%)')
plt.show()