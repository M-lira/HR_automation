from sqlalchemy import create_engine, Column, Integer, String, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create a Base class
Base = declarative_base()

class Employee(Base):
    __tablename__ = 'employees'

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String, unique=True, index=True)
    name = Column(String)
    department = Column(String)
    start_date = Column(Date, nullable=True)
    termination_date = Column(Date, nullable=True)
    last_training_date = Column(Date, nullable=True)
    email = Column(String, nullable=True)

class Training(Base):
    __tablename__ = 'trainings'

    id = Column(Integer, primary_key=True, index=True)
    available_training = Column(String, unique=True)

DATABASE_URL = "sqlite:///./Hr_automation.db" 

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables in the database
def init_db():
    Base.metadata.create_all(bind=engine)
