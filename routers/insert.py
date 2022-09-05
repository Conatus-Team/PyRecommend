"""
DB에 데이터 insert 하는 코드
embedding 계산이 오래 걸려서 계산 후 DB에 저장
나중에 사용할 때는 데이터 불러와서 사용
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from conn.db_class import SmallTalk


import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import urllib.request
from sentence_transformers import SentenceTransformer


router = APIRouter(
    prefix="/insert",
    tags=["insert"],
    responses={404: {"description": "Not found"}},
)

@router.post("/")
async def first_post(request: Request):
    # DB 세션
    session = request.state.db_conn

    # 기존 데이터 삭제
    original_data = session.query(SmallTalk).delete()
    session.commit()
    
    # DB에 임베딩 미리 저장시키는 로직
    # 학습 데이터 불러오기
    train_data = urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatBotData.csv")
    # train_data = pd.read_csv('ChatBotData.csv')
    train_data = pd.read_csv('ipynb/ChatBotData.csv')
    print("read csv 완료")
    print(train_data.head())

    # 사전 훈련된 BERT를 로드
    # 한국어도 포함되어 학습된 다국어 모델을 로드
    model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
    print("model 로드 완료")

    # 모든 질문열. 즉, train_data['Q']에 대해서 문장 임베딩 값을 구한 후 embedding이라는 새로운 열에 저장
    # 계산이 좀 오래 걸림 -> DB에 train_data를 저장 (임베딩 저장 목적)
    train_data['embedding'] = train_data.apply(lambda row: model.encode(row.Q), axis = 1)

    print("임베딩 계산 완료")  
    print(len(train_data['embedding'][0].tolist()))


    # DB 저장할 준비
    for idx, value in train_data.iterrows():
        question = value['Q']
        answer = value['A']
        embedding = value['embedding'].tolist()
        addMemo = SmallTalk(question=question, answer= answer, embedding=embedding)
        session.add(addMemo)

    # DB 저장
    session.commit()
    print("finished")
    return "finished"