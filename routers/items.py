"""
라우터 만들기 연습 파일
"""


from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import List
from conn.db_class import Test, SmallTalk
# from ..dependencies import get_token_header

# 라우터 정보 기입
router = APIRouter(
    prefix="/items",
    tags=["items"],
    # dependencies=[Depends(get_token_header)],
    responses={404: {"description": "Not found"}},
)

# 데이터 모델
class Item(BaseModel):
    name : str
    number : int
    json_data : List[int]

class QA(BaseModel):
    Q : str
    A: str
    embedding: List[float]


# 더미데이터
fake_items_db = {"plumbus": {"name": "Plumbus"}, "gun": {"name": "Portal Gun"}}

@router.get("/")
async def read_items():
    return fake_items_db

# path variable 읽기
# http://localhost:8000/items/plumbus
@router.get("/{item_id}")
async def read_item(item_id: str):
    if item_id not in fake_items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"name": fake_items_db[item_id]["name"], "item_id": item_id}

# query string 읽기
# http://localhost:8000/items/name/?item_id=plumbus
@router.get("/name/")
async def read_item(item_id: str):
    if item_id not in fake_items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"name": fake_items_db[item_id]["name"], "item_id": item_id}

@router.put(
    "/{item_id}",
    tags=["custom"],
    responses={403: {"description": "Operation forbidden"}},
)
async def update_item(item_id: str):
    if item_id != "plumbus":
        raise HTTPException(
            status_code=403, detail="You can only update the item: plumbus"
        )
    return {"item_id": item_id, "name": "The great Plumbus"}


@router.post("/")
async def post_item(item_id: str):
    if item_id not in fake_items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"name": fake_items_db[item_id]["name"], "item_id": item_id}

# 디비 쿼리를 위한 종속성 주입을 위한 함수
def get_db_conn(request: Request):
    return request.state.db_conn		# middleware 에서 삽입해준 db_conn

# DB 연결
@router.post("/post")
async def first_post(request: Request, item:Item):
    addMemo = Test(name=item.name, number=item.number, json_data=item.json_data)
    session = request.state.db_conn
    session.add(addMemo)
    session.commit()
    print("add new item")
    return item


@router.post("/post_train")
async def first_post(request: Request):
    # addMemo = {
    #     "Q": "놀자",
    #     "A" : "그래",
    #     "embedding": [2.3, 2.4, 6.3]
    # }
    addMemo = SmallTalk(Q="놀자", A="그래", embedding=[1, 2, 3])
    session = request.state.db_conn
    session.add(addMemo)
    session.commit()
    print("add new item")
    return addMemo
        