from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import Settings, get_settings
from fastapi import Depends

# Create a Declarative Meta instance
Base = declarative_base()

# DB Dependency
def get_db(settings: Settings = Depends(get_settings)):
   
    # Create engine
    engine = create_engine(f'mssql+pymssql://{settings.DB_UID}:{settings.DB_PWD}@{settings.DB_SERVER}/{settings.DB_NAME}')
    
    # Create Session
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()