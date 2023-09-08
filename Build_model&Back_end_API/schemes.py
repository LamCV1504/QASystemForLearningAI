from pydantic import BaseModel


class Data(BaseModel):
  text: str
  label1: str
  label2: str
  id_key: int

  class Config:
    orm_mode = True