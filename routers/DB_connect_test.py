from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import List
from conn.db_class import Test, SmallTalk

class QA(BaseModel):
    Q : str
    A: str
    embedding: List[float]

router = APIRouter(
    prefix="/db",
    tags=["db"],
    responses={404: {"description": "Not found"}},
)

@router.post("/post")
async def first_post(request: Request):
    # DB 세션
    session = request.state.db_conn
    addMemo = SmallTalk(question="q", answer= "a", embedding=[1, 2, 3])
    session.add(addMemo)
    session.commit()