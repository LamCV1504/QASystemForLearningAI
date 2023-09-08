
import models

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from connection import get_db
from predict_label import predict_final
from get_good_answers import tfidf_cosineSimilarity

app = FastAPI()
origins = [ '*' ]

app.add_middleware(CORSMiddleware, allow_origins= origins, allow_credentials = True, allow_methods = ["*"], allow_headers = ['*'])

@app.get('/')
async def get_answer(question: str, db: Session = Depends(get_db)):

  question, label = predict_final(question, 2)

  if len(question) == 0:
    return {"errorCode": 1, "answer": 'Hệ thống không hiểu câu hỏi, vui lòng nhập câu hỏi rõ ràng hơn!'}

  queryData = db.query(models.Data).filter(models.Data.label2 == label)
  data = queryData.all()

  data_texts = [row.text for row in data]

  answer = tfidf_cosineSimilarity(question, data_texts)

  return {"errorCode": 0, "answer": answer}
