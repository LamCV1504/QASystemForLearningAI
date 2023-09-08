
from sqlalchemy import  NVARCHAR, Integer, Column
from connection import Base

class Data(Base):
    __tablename__ = 'data'

    text = Column(NVARCHAR(max))
    label1 = Column(NVARCHAR(max))
    label2 = Column(NVARCHAR(max))
    id_key = Column(Integer, primary_key = True, autoincrement=True)
