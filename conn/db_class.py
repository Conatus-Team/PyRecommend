from sqlalchemy import Column, TEXT, BIGINT, TIMESTAMP, INTEGER, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.types import JSON
from sqlalchemy.dialects.mssql import TINYINT
Base = declarative_base()


# hobby
class Hobby(Base):
    __tablename__ = 'hobby'
    id = Column(BIGINT,nullable=False, autoincrement=True, primary_key=True)
    name = Column(String)
    place = Column(INTEGER)
    compete = Column(INTEGER)
    strategic = Column(INTEGER)
    group = Column(INTEGER)
    type = Column(INTEGER)

class HobbyPreference(Base):
    __tablename__ = 'hobby_preference'
    id = Column(BIGINT,nullable=False, autoincrement=True, primary_key=True)
    user_id = Column(BIGINT)
    tag = Column(String)
    count = Column(INTEGER)


class UserHobby(Base):
    __tablename__ = 'user_hobby'
    id = Column(BIGINT,nullable=False, autoincrement=True, primary_key=True)
    user_id = Column(BIGINT)
    hobby_id = Column(BIGINT)
    hobby_name = Column(String)


# group


# lecture
# class UserHobby(Base):
#     __tablename__ = 'user_hobby'
#     id = Column(BIGINT,nullable=False, autoincrement=True, primary_key=True)
#     user_id = Column(BIGINT)
#     hobby_id = Column(BIGINT)
#     hobby_name = Column(String)