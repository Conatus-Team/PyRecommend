from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
import json
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SECRET_FILE = os.path.join(BASE_DIR, 'secrets.json')
secrets = json.loads(open(SECRET_FILE).read())
app = secrets["DB"]


# app = {
#         'name': 'mysql+pymysql',
#         'user': 'root',
#         'password': '1234',
#         'host': 'cb_db',
#         'dbconn': 'test_db',
#         'port': '3306'
#     }

conn_string=f'{app["name"]}://{app["user"]}:{app["password"]}@{app["host"]}:{app["port"]}/{app["dbconn"]}'

class engineconn:

    def __init__(self):
        self.engine = create_engine(conn_string, pool_recycle =500, encoding = 'utf-8')

    def sessionmaker(self):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        return session

    def connection(self):
        conn = self.engine.connect()
        print()
        print("-----------------------------------------")
        print("--------------DB CONNECTED---------------")
        print("-----------------------------------------")
        print()
        return conn