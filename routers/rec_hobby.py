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
# from ..dependencies import get_token_header

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

# 라우터 정보 기입
router = APIRouter(
    prefix="/recommend/hobby",
    tags=["hobby"],
    # dependencies=[Depends(get_token_header)],
    responses={404: {"description": "Not found"}},
)


from sklearn.metrics.pairwise import cosine_similarity
import operator
# collaborative filtering에서 사용
# current_user와 가장 유사한 사용자를 찾는 함수
# k : 결과로 리턴할 비슷한 유저의 수
def similar_users(user_id, matrix, k=3):
    # print(matrix)
    # create a df of just the current user
    user = matrix[matrix.index == user_id]
    
    # and a df of all other users
    other_users = matrix[matrix.index != user_id]
    
    # calc cosine similarity between user and each other user
    similarities = cosine_similarity(user,other_users)[0].tolist()
    
    # create list of indices of these users
    indices = other_users.index.tolist()
    
    # create key/values pairs of user index and their similarity
    index_similarity = dict(zip(indices, similarities))
    
    # sort by similarity
    index_similarity_sorted = sorted(index_similarity.items(), key=operator.itemgetter(1))
    index_similarity_sorted.reverse()
    
    # grab k users off the top
    top_users_similarities = index_similarity_sorted[:k]
    # print(top_users_similarities) # [('12', 0.5736034008043722), ('5', 0.4905516816553172), ('6', 0.4782851787874137)]
    users = [u[0] for u in top_users_similarities]
    
    return users

# current_user = '10'
# # try it out
# similar_user_indices = similar_users(current_user, rating_matrix)
# print(similar_user_indices) # ['12', '5', '6']

# collaborative filtering 취미 추천 함수
# items: 결과적으로 추천할 취미의 수
def recommend_item(user_index:String, similar_user_indices, matrix, user_hobby, items=5):
    
    # 비슷한 사람 찾기
    # load vectors for similar users
    similar_users = matrix[matrix.index.isin(similar_user_indices)]
    # calc avg ratings across the 3 similar users
    similar_users = similar_users.mean(axis=0)
    
    # convert to dataframe so its easy to sort and filter
    similar_users_df = pd.DataFrame(similar_users, columns=['mean'])
    # print(f"similar users:")
    # print(similar_users_df.head())
    # load vector for the current user
    user_df = matrix[matrix.index == user_index]
    # transpose it so its easier to filter
    user_df_transposed = user_df.transpose()
    # rename the column as 'rating'
    user_df_transposed.columns = ['rating']
    # remove any rows without a 0 value. Anime not watched yet
    user_df_transposed = user_df_transposed[user_df_transposed['rating']==0]
    # # generate a list of animes the user has not seen
    # animes_unseen = user_df_transposed.index.tolist()
    
    # user_hobby
    # print(user_hobby.head())
    selected_hobby = user_hobby[user_hobby["user_id"] == user_index].hobby.tolist()

    
    # filter avg ratings of similar users for only anime the current user has not seen
    # ~; is not in 문법
    similar_users_df_filtered = similar_users_df[~similar_users_df.index.isin(selected_hobby)]

    # order the dataframe
    similar_users_df_ordered = similar_users_df_filtered.sort_values(by=['mean'], ascending=False)

    # grab the top n anime   
    top_n_hobby = similar_users_df_ordered.head(items)
    # print(top_n_hobby)
#                  mean
# hobby            
# 독서     275.333333
# 토론     234.333333
# 농구     208.666667
# 요리     155.000000
# 등산     141.666667
    top_n_hobby_indices = top_n_hobby.index.tolist()

    return top_n_hobby_indices


# Content Based Recommend
def genre_recommendations(target_title, matrix, items, k=10):
    recom_idx = matrix.loc[:, target_title].values.reshape(1, -1).argsort()[:, ::-1].flatten()[1:k+1]
    recom_title = items.iloc[recom_idx, :].hobby.values
    recom_genre = items.iloc[recom_idx, :]['type'].values
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
@router.get("/")
async def recommend_hobby(request: Request):
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

    # ===================================
    # =     collaborative filtering     =
    # ===================================

    # user_hobby_rating 기록하기 위한 dataframe 만들기
    user_hobby_rating = pd.DataFrame(user_hobby, columns = ['user_id', 'hobby', 'rating'])
    user_hobby_rating = user_hobby_rating.fillna(10)

    # 본인이 가지고 있는 취미는 10점 부여
    user_hobby_rating = pd.DataFrame(user_hobby, columns = ['user_id', 'hobby', 'rating'])
    user_hobby_rating = user_hobby_rating.fillna(10)
    # user_hobby_rating.head()

    print("=======================")
    print(f"user_hobby: {user_hobby.shape}")
    print(user_hobby_rating.shape)
    print(user_hobby.head())
    print(user_hobby_rating.head())
    print("=======================")

    # user_lecture 정보를 이용해 rating 계산
    for idx, value in user_lecture.iterrows():
        hobby = value['hobby']
        user_id = value['user_id']
        is_liked = value['is_liked']
        click_count = value['click_count']
        score = float(is_liked) * 5 + float(click_count)
        if (len(user_hobby_rating[(user_hobby_rating['hobby'] ==hobby) & (user_hobby_rating['user_id'] == user_id)]) == 1):
            # print(user_hobby_rating[(user_hobby_rating['hobby'] ==hobby) & (user_hobby_rating['user_id'] == user_id)])
            user_hobby_rating.loc[(user_hobby_rating['hobby'] ==hobby) & (user_hobby_rating['user_id'] == user_id), 'rating'] += score
            # user_hobby_rating.loc[(user_hobby_rating['hobby'] ==hobby) & (user_hobby_rating['user_id'] == user_id)] += score
        else:
            new_data = {
                'user_id' : user_id,
                'hobby' : hobby,
                'rating' : score
            }
            user_hobby_rating = user_hobby_rating.append(new_data, ignore_index=True)
    print("=======================")
    print(f"user lecture: {user_lecture.shape}")
    print(user_hobby_rating.shape)
    print("=======================")

    # group_activity 아용해 rating 계산
    for idx, value in group_activity.iterrows():
        hobby = value['hobby']
        user_id = value['user_id']
        registered = value['registered']
        upload_pictures = value['upload_pictures']
        upload_content = value['upload_content']
        clicked = value['clicked']
        score = float(registered) * 3 + float(upload_pictures) * 6 + float(upload_content) * 6 + float(clicked)
        if (len(user_hobby_rating[(user_hobby_rating['hobby'] ==hobby) & (user_hobby_rating['user_id'] == user_id)]) == 1):
            # print(user_hobby_rating[(user_hobby_rating['hobby'] ==hobby) & (user_hobby_rating['user_id'] == user_id)])
            user_hobby_rating.loc[(user_hobby_rating['hobby'] ==hobby) & (user_hobby_rating['user_id'] == user_id), 'rating'] += score
            # user_hobby_rating.loc[(user_hobby_rating['hobby'] ==hobby) & (user_hobby_rating['user_id'] == user_id)] += score
        else:
            new_data = {
            'user_id' : user_id,
            'hobby' : hobby,
            'rating' : score
            }
            user_hobby_rating = user_hobby_rating.append(new_data, ignore_index=True)
    print("=======================")
    print(f"group activity: {group_activity.shape}")
    print(user_hobby_rating.shape)
    print("=======================")

    # ~~~~~~~~ Feedback ~~~~~~~~~~
    # hobby_preference 이용해 rating 계산

    for idx, value in hobby_preference.iterrows():

        user_id = str(value['user_id'])
        tag = value['tag']
        count = value['count']
  
        tag_name, tag_num = tag.split('_')
        # print(tag_name)
        # print(tag_num)
        # print(hobby_list.head())
        score = float(count) * 3

        hobbies = hobby_list.loc[(hobby_list[tag_name] == int(tag_num)), 'hobby']
        # print(len(hobbies))
        # break
        for hobby in hobbies:
            if (len( user_hobby_rating[(user_hobby_rating['hobby'] ==hobby) & (user_hobby_rating['user_id'] == user_id)]) == 1):
                # print(user_hobby_rating[(user_hobby_rating['hobby'] ==hobby) & (user_hobby_rating['user_id'] == user_id)])
                user_hobby_rating.loc[(user_hobby_rating['hobby'] ==hobby) & (user_hobby_rating['user_id'] == user_id), 'rating'] += score
                # user_hobby_rating.loc[(user_hobby_rating['hobby'] ==hobby) & (user_hobby_rating['user_id'] == user_id)] += score
            else:
                new_data = {
                    'user_id' : user_id,
                    'hobby' : hobby,
                    'rating' : score
                }
                user_hobby_rating = user_hobby_rating.append(new_data, ignore_index=True)

    print("=======================")
    print(f'hobby_preference: {hobby_preference.shape}')
    print(user_hobby_rating.shape)
    print("=======================")


    # rating matrix 만들기 (user_based)
    rating_matrix = user_hobby_rating.pivot_table(index='user_id', columns='hobby', values='rating')
    # replace NaN values with 0
    rating_matrix = rating_matrix.fillna(0)


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

    result_dic = dict()
    result_dic["data"] = []

    # user_id string으로 바꾸기
    user_id_list = list(map(str, sorted(list(user_id_list))))
    # 타겟 유저 설정
    for target_user in user_id_list:
        # collaborative filtering
        similar_user_indices = similar_users(target_user, rating_matrix, 5)
        user_based_result = recommend_item(target_user, similar_user_indices, rating_matrix, user_hobby, 10)

        # contents based filtering
        best_hobby = user_hobby_rating[user_hobby_rating['user_id'] == target_user].sort_values(by='rating', ascending = False).head(1)["hobby"].values[0]
        contents_based_result = genre_recommendations(best_hobby, cosine_sim_df, hobby_list)
        contents_based_result = contents_based_result["recom_title"].values

        result = []
        i = 0
        while len(result) < 10:
            if not (user_based_result[i] in result):
                result.append(user_based_result[i])
            if not (contents_based_result[i] in result):
                result.append(contents_based_result[i])
            i += 1
        result = result[:10]
        tmp = {"id": int(target_user), "recommend": result}
        result_dic["data"].append(tmp)

    return (result_dic)

    # return "get ressponse"

    

    