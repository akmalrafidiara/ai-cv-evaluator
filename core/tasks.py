import json
from celery import Celery
from .config import settings
from .database import SessionLocal, EvaluationJob, Document
from datetime import datetime

from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

celery_app = Celery(
  'ai_cv_evaluator',
  broker=settings.CELERY_BROKER_URL,
  backend=settings.CELERY_RESULT_BACKEND
)
celery_app.conf.update(
  task_track_started=True,
  task_default_retry_delay=5,
  task_max_retries=3
)

client = genai.Client(api_key=settings.GEMINI_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, google_api_key=settings.GEMINI_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=settings.GEMINI_API_KEY)

# QDRANT_URL = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
# COLLECTION_NAME = "ai_evaluation_context"

QDRANT_HOST = settings.QDRANT_HOST 
QDRANT_PORT = settings.QDRANT_PORT
COLLECTION_NAME = "ai_evaluation_context" 

qdrant_native_client = QdrantClient(
    host=QDRANT_HOST,
    port=6333,
    prefer_grpc=True
)

qdrant_client = Qdrant(
    client=qdrant_native_client,
    embeddings=embeddings, 
    collection_name=COLLECTION_NAME
)

def parse_pdf_to_text(file_path: str) -> str:
  loader = PyPDFLoader(file_path)
  pages = loader.load()
  return "\n".join([page.page_content for page in pages])

def retrieve_context(query: str, doc_type: str) -> str:
  qdrant_filter = Filter(
      must=[
          FieldCondition(key="type", match=MatchValue(value=doc_type))
      ]
  )

  search_result = qdrant_client.similarity_search(
    query=query,
    k=4,
    filter=qdrant_filter
  )

  return "\n--\n".join([doc.page_content for doc in search_result])

@celery_app.task(bind=True)
def run_evaluation_pipeline(self, job_id: str):
  print(f"Starting evaluation pipeline for job {job_id}")

  db = SessionLocal()
  try:
    job = db.query(EvaluationJob).filter(EvaluationJob.job_id == job_id).first()
    if not job:
      print(f"Job {job_id} not found")
      return
    
    job.status = "processing"
    db.commit()

    # load CV and report documents
    cv_doc = db.query(Document).filter(Document.id == job.cv_document_id).first()
    report_doc = db.query(Document).filter(Document.id == job.report_document_id).first()

    # Pharse CV and report to text
    cv_text = parse_pdf_to_text(cv_doc.file_path)
    report_text = parse_pdf_to_text(report_doc.file_path)

    print("-> Running CV Evaluation (Stage 1)")
    cv_context_jd = retrieve_context(cv_text, "job_description")
    cv_context_rubric = retrieve_context("Scoring Rubric for CV Match", "scoring_rubric")

    cv_prompt_template = """
        Anda adalah analisis dan evaluator ahli Human Resource. Tugas Anda adalah menilai kecocokan CV kandidat ini.
        INSTRUKSI UTAMA: Berikan output dalam format JSON yang valid. Jangan ada teks tambahan.
        
        JOB DESCRIPTION: {job_description}
        SCORING RUBRIC: {scoring_rubric}
        CANDIDATE CV: {cv_content}
        
        Evaluasi kriteria: Technical Skills (40%), Experience Level (25%), Relevant Achievements (20%), Cultural Fit (15%).
        Hitung weighted average dari skor 1-5 berdasarkan rubrik, lalu konversi ke float 0.0-1.0 (skor rata-rata * 0.2).
        
        FORMAT OUTPUT (JSON): 
        {{"cv_match_rate": <float 0.0-1.0>, "cv_feedback": "<Ringkasan evaluasi, 3-4 kalimat>}}
        """
    
    cv_prompt = PromptTemplate(
          input_variables=["job_description", "scoring_rubric", "cv_content"],
          template=cv_prompt_template
      )
    
    cv_chain = LLMChain(llm=llm, prompt=cv_prompt)

    cv_output_raw = cv_chain.invoke({
            "job_description": cv_context_jd,
            "scoring_rubric": cv_context_rubric,
            "cv_content": cv_text
        })['text']
    # print(f"DEBUG LLM RAW OUTPUT (CV): {cv_output_raw}")
    # cv_result = json.loads(cv_output_raw)

    try:
        json_start = cv_output_raw.find('{')
        json_end = cv_output_raw.rfind('}')
        
        if json_start == -1 or json_end == -1:
            raise ValueError("LLM response did not contain a valid JSON block.")

        clean_json_string = cv_output_raw[json_start : json_end + 1]
        cv_result = json.loads(clean_json_string)
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"DEBUG: Failed to parse Project JSON. Raw Output: {cv_output_raw}")
        raise self.retry(exc=e)

    print("-> Running Project Report Evaluation (Stage 2)")
    report_context_brief = retrieve_context(report_text, "case_study")
    report_context_rubric = retrieve_context("Scoring Rubric for Project Deliverables", "scoring_rubric")

    report_prompt_template = """
        Anda adalah reviewer teknis. Nilai Project Report ini berdasarkan Case Study Brief dan Rubrik.
        INSTRUKSI UTAMA: Berikan output dalam format JSON yang valid. Jangan ada teks tambahan.
        
        CASE STUDY BRIEF: {case_study_brief}
        SCORING RUBRIC: {scoring_rubric}
        PROJECT REPORT CONTENT: {report_content}
        
        Evaluasi kriteria: Correctness (30%), Code Quality (25%), Resilience (20%), Documentation (15%), Creativity (10%). Hitung weighted average dari skor 1-5.
        
        FORMAT OUTPUT (JSON): 
        {{"project_score": <float 1.0-5.0>, "project_feedback": "<Ringkasan evaluasi teknis, 3-4 kalimat>}}
        """
    report_prompt = PromptTemplate(
        input_variables=["case_study_brief", "scoring_rubric", "report_content"],
        template=report_prompt_template
    )

    report_chain = LLMChain(llm=llm, prompt=report_prompt)
    report_output_raw = report_chain.invoke({
        "case_study_brief": report_context_brief,
        "scoring_rubric": report_context_rubric,
        "report_content": report_text
    })['text']
    
    try:
        json_start = report_output_raw.find('{')
        json_end = report_output_raw.rfind('}')
        
        if json_start == -1 or json_end == -1:
            raise ValueError("LLM response did not contain a valid JSON block.")

        clean_json_string = report_output_raw[json_start : json_end + 1]
        report_result = json.loads(clean_json_string)
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"DEBUG: Failed to parse Project JSON. Raw Output: {report_output_raw}")
        raise self.retry(exc=e)

    print("-> Running Final Synthesis (Stage 3)")
    final_prompt_template = """
      Sintesiskan hasil evaluasi CV dan Proyek menjadi ringkasan kandidat 3-5 kalimat.
      Fokus pada: 1. Kekuatan utama (strengths). 2. Kesenjangan/kelemahan (gaps). 3. Rekomendasi tindak lanjut.
      
      CV MATCH RATE: {match_rate}
      CV FEEDBACK: {cv_feedback}
      PROJECT SCORE: {score}
      PROJECT FEEDBACK: {report_feedback}
      
      OVERALL SUMMARY: 
    """
    final_prompt = PromptTemplate(
        input_variables=["match_rate", "cv_feedback", "score", "report_feedback"],
        template=final_prompt_template
    )

    final_chain = LLMChain(llm=llm, prompt=final_prompt)
    overall_summary = final_chain.invoke({
        "match_rate": cv_result["cv_match_rate"],
        "cv_feedback": cv_result["cv_feedback"],
        "score": report_result["project_score"],
        "report_feedback": report_result["project_feedback"]
    })['text']

    final_result = {
            "cv_match_rate": cv_result["cv_match_rate"],
            "cv_feedback": cv_result["cv_feedback"],
            "project_score": report_result["project_score"],
            "project_feedback": report_result["project_feedback"],
            "overall_summary": overall_summary
        }

    job.status = "completed"
    job.result_data = final_result
    job.completed_at = datetime.utcnow()
    db.commit()

    print(f"Completed evaluation pipeline for job {job_id}")

  except Exception as e:
    print(f"FATAL ERROR in job {job_id}: {str(e)}. Attempting retry.")
    db.rollback() 
    
    if self.request.retries < celery_app.conf.task_max_retries:
        raise self.retry(exc=e)
    else:
        job.status = "failed"
        job.completed_at = datetime.utcnow()
        db.commit()
        print(f"Job {job_id} failed permanently after max retries.")
  finally:
    db.close()
