"""
취미 추천
"""


from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import List
from conn.db_class import *
import pandas as pd
import json
import numpy as np


import random

# from ..dependencies import get_token_header

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

# 라우터 정보 기입
router = APIRouter(
    prefix="/recommend/hobby/contents",
    tags=["hobby-contents-based"],
    # dependencies=[Depends(get_token_header)],
    responses={404: {"description": "Not found"}},
)


# Content Based Recommend
def genre_recommendations(target_title, matrix, items, k=10):
    print(target_title)
    recom_idx = matrix.loc[:, target_title].values.reshape(1, -1).argsort()[:, ::-1].flatten()[1:k+1]
    recom_title = items.iloc[recom_idx, :].hobby.values
    print(recom_title)
    recom_genre = items.iloc[recom_idx, :]['type'].values
    print(recom_genre)
    target_title_list = np.full(len(range(k)), target_title)
    target_genre_list = np.full(len(range(k)), items[items.hobby == target_title]['type'].values)
    d = {
        'target_title':target_title_list,
        'target_genre':target_genre_list,
        'recom_title' : recom_title,
        'recom_genre' : recom_genre
    }
    return pd.DataFrame(d)

# contents based에서 사용
# 원핫 데이터로 만들어주는 함수
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
from sklearn.metrics.pairwise import cosine_similarity
import operator
"""
target_column: 해당 컬럼을 one-hot 데이터로 만들어줌
use_number_column: 새로 생긴 컬럼 이름을 원래 값이 아닌 숫자로 붙여줌
  예시: True일때-> fruit_1  False일때 -> fruit_apple
"""
def onehot_endcode(target_df, target_column, hobby_list, use_number_column = True):
  #reshape the 1-D country array to 2-D as fit_transform expects 2-D and finally fit the object 
  X = onehotencoder.fit_transform(hobby_list[target_column].values.reshape(-1,1)).toarray()
  #To add this back into the original dataframe 

  # 새로 생기는 칼럼 이름
  col_name = [target_column+"_"+str(i) for i in (hobby_list[target_column].unique())]
  if (use_number_column): 
    col_name = [target_column+"_"+str(i) for i in range(len(hobby_list[target_column].unique()))]

  # 원핫 데이터를 df로 만들기
  dfOneHot = pd.DataFrame(X, columns = col_name)

  # 원래 df에 이어붙이기
  df = pd.concat([hobby_list, dfOneHot], axis=1)
  
  #droping the target_column
  df= df.drop([target_column], axis=1) 
  #printing to verify 
#   print(df.head())
  return df


# # try it out
# user_based_result = recommend_item('1', similar_user_indices, rating_matrix)

# DB 연결
@router.get("/{target_user_id}")
async def recommend_hobby(request: Request, target_user_id: int):
    print(target_user_id)
    db_conn = request.state.db_conn
    
    # db 읽고 pandas dataframe으로 만들기
    hobby_list = pd.read_sql("hobby", db_conn).drop(columns=["created_time", "updated_time"])
    hobby_list.rename(columns = {'name':'hobby'},inplace=True)

    hobby_preference = pd.read_sql("hobby_preference", db_conn).drop(columns=["created_time", "updated_time"])

    user_hobby = pd.read_sql("user_hobby", db_conn).drop(columns=["created_time", "updated_time"])
    user_hobby = user_hobby.drop(["hobby_id"], axis=1) 
    user_hobby.rename(columns = {'hobby_name':'hobby'},inplace=True)

    group_activity = pd.read_sql("user_group", db_conn).drop(columns=["created_time", "updated_time"])

    user_lecture = pd.read_sql("user_lecture", db_conn).drop(columns=["created_time", "updated_time"])

    #  유저 id 목록, 유저 수 계산
    # lecture_user = user_lecture.iloc[:, 1]
    # hobby_user = user_hobby.iloc[:, 1]
    # group_user = group_activity.iloc[:, 0]
    # print(lecture_user)
    #  유저 id 목록, 유저 수 계산

    lecture_user = user_lecture.loc[:, "user_id"]
    hobby_user = user_hobby.loc[: ,"user_id"]
    group_user = group_activity.loc[: , "user_id"]

    user_id_list = set(map(int, list(lecture_user))).union(set(map(int, list(hobby_user)))).union(set(map(int, list(group_user))))
    user_count = len(user_id_list)
    print(f"user_count: {user_count}")
    print(f"======================")
    print(f"user_id_list: {user_id_list}")

    db_conn.close()

    # ===================================
    # =    Contents based filtering     =
    # ===================================


    # ont-hot
    data = hobby_list.iloc[:, 1:]

    data = onehot_endcode(data, "type", hobby_list)
    data.set_index('hobby',inplace = True)



    # cosine similarity
    cosine_sim = cosine_similarity(data)
    cosine_sim_df = pd.DataFrame(cosine_sim, index = data.index, columns = data.index)

    # ===================================
    # =          최종 추천              =
    # ===================================

    result_list = list()
    # result_dic["data"] = []

    ## user_id string으로 바꾸기 -> int로 해야 함
    # user_id_list = list(map(str, sorted(list(user_id_list))))
    # user_id_list = list(map(int, sorted(list(user_id_list))))


    user_id_list = [target_user_id]
    # 타겟 유저 설정
    for target_user in user_id_list:
        # contents based filtering
        print(target_user)
        best_hobby = user_hobby[user_hobby['user_id'] == target_user]
        best_hobby = best_hobby["hobby"].values
        print(best_hobby)
        # best_hobby = user_hobby_rating[user_hobby_rating['user_id'] == target_user].sort_values(by='rating', ascending = False).head(1)["hobby"].values[0]
        
        # best_hobby에서 랜덤 3개 뽑기
        target_best_hobby = [ best_hobby[random.randrange(0,len(best_hobby))] for _ in range(3)]
        print(target_best_hobby)
        result = []

        for best_hobby_item in target_best_hobby:
            print(best_hobby_item)
            contents_based_result = genre_recommendations(best_hobby_item, cosine_sim_df, hobby_list, k=6)
            print(contents_based_result)
            contents_based_result = contents_based_result["recom_title"].values
            for rec_hobby in contents_based_result:
                if rec_hobby not in target_best_hobby and  rec_hobby not in result:
                    result.append(rec_hobby)
                    break

        # result = []
        # i = 0
        # while len(result) < 10:
        #     if not (user_based_result[i] in result):
        #         result.append(user_based_result[i])
        #     if not (contents_based_result[i] in result):
        #         result.append(contents_based_result[i])
        #     i += 1
        # result = result[:10]
        tmp = {"id": int(target_user), "recommend": result}
        result_list.append(tmp)

    return (result_list)

    # return "get ressponse"

    

    