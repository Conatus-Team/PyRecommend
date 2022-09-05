from sqlalchemy import Column, TEXT, INT, BIGINT, TIMESTAMP
from sqlalchemy.orm import declarative_base
from sqlalchemy.types import JSON
from sqlalchemy.dialects.mssql import TINYINT
Base = declarative_base()



"""
-- test_db.test definition

CREATE TABLE `test` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `name` text NOT NULL,
  `number` int NOT NULL,
  `json_data` json DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=9 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
"""

class Test(Base):
    __tablename__ = 'test'
    id = Column(BIGINT,nullable=False, autoincrement=True, primary_key=True)
    name = Column(TEXT, nullable=False)
    number = Column(INT, nullable=False)
    json_data = Column(JSON)


"""
-- test_db.small_talk definition

CREATE TABLE `small_talk` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `Q` text NOT NULL,
  `A` text NOT NULL,
  `embedding` json NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `is_deleted` tinyint(1) NOT NULL DEFAULT '0',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=47302 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
"""
class SmallTalk(Base):
    __tablename__ = 'chat_train'
    id = Column(BIGINT,nullable=False, autoincrement=True, primary_key=True)
    question = Column(TEXT, nullable=False)
    answer = Column(TEXT, nullable=False)
    embedding = Column(JSON, nullable=False)
    # created_at = Column(TIMESTAMP)
    # updated_at = Column(TIMESTAMP)
    # is_deleted = Column(TINYINT)


# # JSON data
# # Read a value
# model.args['task']
# # Read a value, cast to type
# model.args['n'].as_integer()

# # Filter by a value
# db.session.query(Pick).filter(Pick.args['task'] = 'Test Task')