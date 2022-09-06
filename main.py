from fastapi import Depends, FastAPI, Request
from pydantic import BaseModel
from conn.db_class import *
from conn.db_conn import engineconn
from sqlalchemy import select
from typing import List

from routers import rec_hobby

app = FastAPI()
app.include_router(rec_hobby.router)
# app.include_router(items.router)
# app.include_router(insert.router)
# app.include_router(chat.router)
# app.include_router(DB_connect_test.router)

engine = engineconn()
session = engine.sessionmaker()


print("======================================")
print("=        SERVER ON PORT 8000         =")
print("======================================")


class Item(BaseModel):
    name : str
    number : int
    json_data : List[int]



@app.get("/")
def first_get():
    return "hello world!!"

@app.get("/get")
async def first_get():
    example = session.execute(
    select(Hobby)
    ).scalars().all()
    print(example[0])
    print(example[0].name)
    return example

# @app.post("/post")
# async def first_post(item:Item):
#     addMemo = Test(name=item.name, number=item.number, json_data=item.json_data)
#     session.add(addMemo)
#     session.commit()
#     print("addMemo")
#     return item

# fastapi middleware, request state 에 db connection 심기
@app.middleware("http")
async def state_insert(request: Request, call_next):
    request.state.db_session = session
    request.state.db_conn = engine.connection()
    response = await call_next(request)
    return response

