from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import uuid

from .config import settings

# Set up the database engine
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Model 
from pydantic import BaseModel

class DocumentIDs(BaseModel):
  """After uploading, return the document IDs."""
  cv_id: str
  report_id: str

class EvaluationJobInput(BaseModel):
  """Input for creating an evaluation job."""
  job_title: str
  cv_id: str
  report_id: str

class JobStatusResponse(BaseModel):
  """Response model for job status."""
  id: str
  status: str

class EvaluationResult(BaseModel):
  """Detail of the evaluation result."""
  cv_match_rate: float
  cv_feedback: str
  project_score: float
  project_feedback: str
  overall_summary: str

class FinalResultResponse(JobStatusResponse):
  """Response model for final evaluation result."""
  result: EvaluationResult

# Table DB models
class Document(Base):
  __tablename__ = "documents"
  id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
  document_type = Column(String, index=True)
  file_path = Column(String, unique=True, index=True)
  uploaded_at = Column(DateTime, default=datetime.utcnow)

class EvaluationJob(Base):
  __tablename__ = "evaluation_jobs"
  job_id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
  cv_document_id = Column(String, nullable=False)
  report_document_id = Column(String, nullable=False)
  job_title = Column(String, nullable=False)
  status = Column(String, default="queued")
  result_data = Column(JSON, nullable=True)
  created_at = Column(DateTime, default=datetime.utcnow)
  completed_at = Column(DateTime, nullable=True)

def create_tables():
  Base.metadata.create_all(bind=engine)

# create_tables()
