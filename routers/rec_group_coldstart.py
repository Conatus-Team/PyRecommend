"""
모임 회원가입한 유저 추천
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
    prefix="/recommend/group/coldstart",
    tags=["group-contents-based"],
    # dependencies=[Depends(get_token_header)],
    responses={404: {"description": "Not found"}},
)


# Content Based Recommend
def genre_recommendations(target_title, matrix, items, k=10):
    # print(target_title)
    # target_title = "디지털 카메라(DSLR) 완전 정복하기"
    recom_idx = matrix.loc[:, target_title].values.reshape(1, -1).argsort()[:, ::-1].flatten()[1:k+1]
    # print(recom_idx)
    # recom_lecture_id = items.iloc[recom_idx, :].lecture_id.values
    # # print(recom_lecture_id)
    recom_title = items.iloc[recom_idx, :].lecture_id.values.tolist()
    # print(recom_title)

    return recom_title
    # recom_genre = items.iloc[recom_idx, :]['category_name'].values
    # # print(recom_genre)
    # target_title_list = np.full(len(range(k)), target_title)
    # target_genre_list = np.full(len(range(k)), items[items.lecture_id == target_title]['category_name'].values)
    # d = {
    #     'target_title':target_title_list,
    #     'target_genre':target_genre_list,
    #     'recom_title' : recom_title,
    #     'recom_genre' : recom_genre
    # }
    # # print(d)
    # return pd.DataFrame(d)

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
  ## printing to verify 
#   # print(df.head())
  return df


# # try it out
# user_based_result = recommend_item('1', similar_user_indices, rating_matrix)

# DB 연결
@router.get("/{target_user_id}")
async def recommend_group_cold_start(request: Request, target_user_id: int):
    # print(target_user_id)
    db_conn = request.state.db_conn
    
    # db 읽고 pandas dataframe으로 만들기
    group_list = pd.read_sql("info", db_conn).drop(columns=["created_time", "updated_time"])
    # print(lecture_list.columns)
    # lecture_list.rename(columns = {'category':'hobby'},inplace=True)

    # hobby_preference = pd.read_sql("hobby_preference", db_conn).drop(columns=["created_time", "updated_time"])

    # user_lecture = pd.read_sql("user_lecture", db_conn).drop(columns=["created_time", "updated_time"])
    # user_lecture = user_lecture.drop(["hobby_id"], axis=1) 
    # user_lecture.rename(columns = {'hobby_name':'hobby'},inplace=True)

    # group_activity = pd.read_sql("user_group", db_conn).drop(columns=["created_time", "updated_time"])

    user_hobby = pd.read_sql("user_hobby", db_conn).drop(columns=["created_time", "updated_time"])
    user_hobby = user_hobby.drop(["hobby_id"], axis=1) 
    user_hobby.rename(columns = {'hobby_name':'hobby'},inplace=True)



    user_group = pd.read_sql("user_group", db_conn).drop(columns=["created_time", "updated_time"])


    # db 연결 해제
    db_conn.close()

    #  유저 id 목록, 유저 수 계산
    # lecture_user = user_lecture.iloc[:, 1]
    # hobby_user = user_hobby.iloc[:, 1]
    # group_user = group_activity.iloc[:, 0]
    # # print(lecture_user)
    #  유저 id 목록, 유저 수 계산

    # lecture_user = user_lecture.loc[:, "user_id"]
    # hobby_user = user_hobby.loc[: ,"user_id"]
    # group_user = group_activity.loc[: , "user_id"]

    # user_id_list = set(map(int, list(lecture_user))).union(set(map(int, list(hobby_user)))).union(set(map(int, list(group_user))))
    # user_count = len(user_id_list)
    # # print(f"user_count: {user_count}")
    # # print(f"======================")
    # # print(f"user_id_list: {user_id_list}")


    # 타겟 유저의 취미
    # print(user_hobby.columns) # Index(['id', 'hobby', 'user_id'], dtype='object')
    target_hobby = user_hobby[user_hobby["user_id"] == target_user_id]["hobby"].values.tolist()
    # print(target_hobby) # ['요리' '십자말풀이' 'DIY' '악기연주']

    # 타겟 유저와 취미가 같은 유저들의 id ( == 비슷한 유저)
    # target_hobby = user_hobby[user_hobby["user_id"] == target_user_id]["hobby"].values
    similar_user = user_hobby[user_hobby.hobby.isin(target_hobby)]
    # 본인 제외
    similar_user = similar_user[similar_user["user_id"] != target_user_id]
    # print(similar_user.head())

    # 비슷한 유저의 id
    similar_user_id = similar_user["user_id"].values



    # 비슷한 유저의 user_lecture
    similar_user_lecture = user_group[user_group.user_id.isin(similar_user_id)]
    # print(similar_user_lecture.head())


    # 비슷한 유저의 lecture rating (찜한 강의는 + 10점, 취미와 같으면 + 5점)
    # user_hobby_rating 기록하기 위한 dataframe 만들기
    user_lecture_rating = pd.DataFrame(similar_user_lecture, columns = ['user_id', 'group_id', 'rating'])
    # print(user_lecture_rating.head())

    # 비슷한 유저가 선택한 강의 id들
    similar_user_lecture_id = list(set(user_lecture_rating.group_id.values.tolist()))
    print(similar_user_lecture_id)

    # 타겟 유저의 취미에 해당하는 강의 id
    target_lecture = group_list[group_list["category"].isin(target_hobby)]
    print(target_lecture.tail())
    target_lecture_id = target_lecture["id"].values.tolist()
    print(target_lecture_id)

    # 타겟 유저의 취미에 해당하는 강의 id를 제외한 비슷한 유저가 선택한 강의 id들
    similar_user_lecture_without_target_lecture = list(set(similar_user_lecture_id) - set(target_lecture_id))

    # 결과 lecture id
    selected_lecture_id = []
    recommend_num = 4 # 추천해야할 개수

    if (len(target_lecture_id)>= recommend_num 
      and len(similar_user_lecture_without_target_lecture) < recommend_num/2):
      selected_lecture_id = random.sample(target_lecture_id, recommend_num)
    elif (len(target_lecture_id)>= 2 and 
          len (similar_user_lecture_without_target_lecture) >= 2):
      selected_lecture_id = random.sample(target_lecture_id, 2) + random.sample(similar_user_lecture_without_target_lecture, 2)
    else:
      left = recommend_num - len(target_lecture_id)
      print(f"left: {left}")
      if (len(similar_user_lecture_without_target_lecture) >= left):
        selected_lecture_id = target_lecture_id+list(random.sample(similar_user_lecture_without_target_lecture, left))
      else:
        selected_lecture_id = similar_user_lecture_without_target_lecture.copy()
    print("+++++++++++++++++++++++++")
    print(selected_lecture_id)

    result_list = list()
    tmp = {"id": int(target_user_id), "recommend": selected_lecture_id}
    result_list.append(tmp)
    return result_list
    # contents based filtering 적용

    # ===================================
    # =    Contents based filtering     =
    # ===================================


    # ont-hot
    data = lecture_list.iloc[:, :]


    data = onehot_endcode(data, "teacher_name", lecture_list)
    data = onehot_endcode(data, "category_name", data)


    data.set_index('lecture_id',inplace = True)

    # print(data.head(1))
    # # print(data.head())


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
        # print(target_user)
        best_hobby = user_lecture[user_lecture['user_id'] == target_user]
        # print(best_hobby.head())
        best_hobby = best_hobby["lecture_id"].values
        # print(best_hobby)
        # best_hobby = user_hobby_rating[user_hobby_rating['user_id'] == target_user].sort_values(by='rating', ascending = False).head(1)["hobby"].values[0]
        
        # best_hobby에서 랜덤 3개 뽑기
        target_best_hobby = [ best_hobby[random.randrange(0,len(best_hobby))] for _ in range(4)]
        result = []

        for best_hobby_item in target_best_hobby:
            contents_based_result = genre_recommendations(best_hobby_item, cosine_sim_df, lecture_list, k=8)
            # # print(contents_based_result)
            # contents_based_result = contents_based_result["recom_title"].values
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
    # print(result_list)

    return result_list

    # return "get ressponse"

    

    