from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from typing import Union
from core.database import SessionLocal, Document, EvaluationJob, DocumentIDs, JobStatusResponse, FinalResultResponse, EvaluationJobInput
from core.tasks import run_evaluation_pipeline
from core.config import settings

app = FastAPI(title="AI CV Evaluator", version="1.0.0")

def get_db():
  db: Session = SessionLocal() 
  try:
    yield db
  finally:
    db.close()

@app.get("/")
async def root():
  return {"message": "API is running."}

@app.post("/upload", response_model=DocumentIDs)
async def upload_files(
  cv_file: UploadFile = File(...),
  report_file: UploadFile = File(...),
  db: Session = Depends(get_db)
):
  storage_dir = "./uploaded_files"
  os.makedirs(storage_dir, exist_ok=True)

  saved_docs = {}

  for file_type, uploaded_file in [("CV", cv_file), ("REPORT", report_file)]:
    if uploaded_file.content_type != "application/pdf":
      raise HTTPException(status_code=400, detail=f"{file_type} must be a PDF file.")
    
    doc_id = str(uuid.uuid4())
    file_location = os.path.join(storage_dir, f"{doc_id}-{file_type}.pdf")

    with open(file_location, "wb") as buffer:
      shutil.copyfileobj(uploaded_file.file, buffer)

    new_doc = Document(id=doc_id, document_type=file_type, file_path=file_location)
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    saved_docs[file_type.lower() + "_id"] = new_doc.id

  return DocumentIDs(cv_id=saved_docs["cv_id"], report_id=saved_docs["report_id"])

@app.post("/evaluate", response_model=JobStatusResponse)
async def trigger_evaluation(
  data: EvaluationJobInput,
  db: Session = Depends(get_db)
):
  cv_doc = db.query(Document).filter(Document.id == data.cv_id, Document.document_type == "CV").first()
  report_doc = db.query(Document).filter(Document.id == data.report_id, Document.document_type == "REPORT").first()

  if not cv_doc or not report_doc:
    raise HTTPException(status_code=404, detail="CV or Report document not found.")
  
  try:
    new_job_id = str(uuid.uuid4())
    new_job = EvaluationJob(
        job_id=new_job_id,
        cv_document_id=data.cv_id,
        report_document_id=data.report_id,
        job_title=data.job_title,
        status="queued",
    )
    db.add(new_job)
    db.commit()

    run_evaluation_pipeline.delay(new_job_id)

    return JobStatusResponse(id=new_job_id, status="queued")

  except Exception as e:
    db.rollback()
    print(f"FATAL ERROR in /evaluate: {e}") 
    raise HTTPException(status_code=500, detail="Internal Server Error: Failed to create or queue job.")


@app.get("/result/{job_id}", response_model=Union[FinalResultResponse, JobStatusResponse]) 
async def get_evaluation_result(
  job_id: str,
  db: Session = Depends(get_db)
):
  job = db.query(EvaluationJob).filter(EvaluationJob.job_id == job_id).first()

  if not job:
    raise HTTPException(status_code=404, detail="Job not found.")
  
  if job.status in ["queued", "processing"]:
    return JobStatusResponse(id=job.job_id, status=job.status)
  
  if job.status == "completed":
    return FinalResultResponse(
      id=job.job_id,
      status=job.status,
      result=job.result_data
    )
  
  return JobStatusResponse(id=job.job_id, status=job.status)

