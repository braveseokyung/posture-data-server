# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime, timedelta
import uuid
import boto3
import os

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, DateTime, ForeignKey
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session as OrmSession

# --- S3 설정 (IAM Role 사용) ---
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "posture-video-bucket")

# IAM Role 덕분에 access key/secret 없이 client 생성 가능
s3 = boto3.client("s3", region_name=AWS_REGION)

# -----------------------------
# DB 설정 (docker-compose 기준 Postgres)
# -----------------------------
# docker-compose.yml에서 설정한 값:
#   POSTGRES_USER: posture
#   POSTGRES_PASSWORD: posture_pw
#   POSTGRES_DB: posture_dev
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://posture:posture_pw@localhost:5433/posture_dev"  # 로컬 개발용 기본값
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db: OrmSession = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -----------------------------
# DB 모델 정의
# -----------------------------
class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, index=True)  # sessionId
    user_id = Column(String, nullable=False)
    device_id = Column(String, nullable=False)
    note = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False)  # 세션 시작 시각(UTC)


class SensorSample(Base):
    __tablename__ = "sensor_samples"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), index=True, nullable=False)
    device_id = Column(String, nullable=False)

    # 세션 시작 기준 offset (ms)
    offset_ms = Column(Integer, nullable=False)

    # 절대 시간 (UTC) — 나중에 Timescale hypertable 만들 때 기준 컬럼
    ts = Column(DateTime, nullable=False, index=True)

    ax = Column(Float, nullable=False)
    ay = Column(Float, nullable=False)
    az = Column(Float, nullable=False)
    gx = Column(Float, nullable=False)
    gy = Column(Float, nullable=False)
    gz = Column(Float, nullable=False)


class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)  # videoId
    session_id = Column(String, ForeignKey("sessions.id"), index=True, nullable=False)
    file_path = Column(String, nullable=False)
    start_offset_ms = Column(Integer, nullable=False)
    fps = Column(Integer, nullable=False)
    duration_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


# 실제 테이블 생성
Base.metadata.create_all(bind=engine)


# -----------------------------
# Pydantic 스키마
# -----------------------------
class SessionCreate(BaseModel):
    userId: str
    deviceId: str
    note: str | None = None


class SessionResponse(BaseModel):
    sessionId: str
    createdAtMs: int


class SensorSampleIn(BaseModel):
    offsetMs: int = Field(..., description="세션 시작 기준 offset (ms)")
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


class SensorBatchIn(BaseModel):
    sessionId: str
    deviceId: str
    samples: List[SensorSampleIn]


class SensorBatchResponse(BaseModel):
    status: str
    sessionId: str
    received: int
    firstOffsetMs: int | None = None
    lastOffsetMs: int | None = None


class HealthResponse(BaseModel):
    status: str
    timeMs: int


class VideoUploadResponse(BaseModel):
    status: str
    videoId: int
    sessionId: str
    filePath: str
    startOffsetMs: int
    fps: int
    durationMs: int | None = None


# -----------------------------
# FastAPI 앱
# -----------------------------
app = FastAPI(title="Posture Data Dev Server (Postgres + ORM)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# 유틸 함수
# -----------------------------
def now_utc() -> datetime:
    return datetime.utcnow()


def now_ms() -> int:
    return int(datetime.utcnow().timestamp() * 1000)


def generate_session_id() -> str:
    return "sess_" + uuid.uuid4().hex[:12]


# -----------------------------
# 엔드포인트들
# -----------------------------
@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", timeMs=now_ms())


# 1) 세션 생성
@app.post("/sessions", response_model=SessionResponse)
def create_session(payload: SessionCreate, db: OrmSession = Depends(get_db)):
    session_id = generate_session_id()
    created_at = now_utc()

    db_session = Session(
        id=session_id,
        user_id=payload.userId,
        device_id=payload.deviceId,
        note=payload.note,
        created_at=created_at,
    )
    db.add(db_session)
    db.commit()

    return SessionResponse(
        sessionId=session_id,
        createdAtMs=int(created_at.timestamp() * 1000),
    )


# 2) 센서 배치 업로드
@app.post("/sensor/batch", response_model=SensorBatchResponse)
def upload_sensor_batch(batch: SensorBatchIn, db: OrmSession = Depends(get_db)):
    # 세션 존재 여부 확인
    session = db.query(Session).filter(Session.id == batch.sessionId).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not batch.samples:
        raise HTTPException(status_code=400, detail="No samples provided")

    base_time: datetime = session.created_at  # 세션 시작 시각 기준

    samples_to_add: list[SensorSample] = []
    for s in batch.samples:
        ts = base_time + timedelta(milliseconds=s.offsetMs)
        samples_to_add.append(
            SensorSample(
                session_id=batch.sessionId,
                device_id=batch.deviceId,
                offset_ms=s.offsetMs,
                ts=ts,
                ax=s.ax,
                ay=s.ay,
                az=s.az,
                gx=s.gx,
                gy=s.gy,
                gz=s.gz,
            )
        )

    db.add_all(samples_to_add)
    db.commit()

    offsets = [s.offsetMs for s in batch.samples]
    return SensorBatchResponse(
        status="ok",
        sessionId=batch.sessionId,
        received=len(batch.samples),
        firstOffsetMs=min(offsets),
        lastOffsetMs=max(offsets),
    )


# 3) 비디오 업로드 (S3에 저장)
@app.post("/videos/upload", response_model=VideoUploadResponse)
async def upload_video(
    sessionId: str = Form(...),
    startOffsetMs: int = Form(...),
    fps: int = Form(...),
    videoFile: UploadFile = File(...)
):
    # 확장자 추출 (없으면 mp4로 기본)
    filename = videoFile.filename or "video.mp4"
    if "." in filename:
        ext = filename.rsplit(".", 1)[-1]
    else:
        ext = "mp4"

    # S3에 저장할 key (폴더 느낌으로 videos/ 밑에 정리)
    file_key = f"videos/{sessionId}_{uuid.uuid4().hex}.{ext}"

    # S3로 업로드
    try:
        # fileobj 기반 업로드
        s3.upload_fileobj(videoFile.file, AWS_S3_BUCKET, file_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 upload failed: {e}")

    # S3 URL 생성
    file_url = f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{file_key}"

    # DB 저장
    db = SessionLocal()
    try:
        video = Video(
            session_id=sessionId,
            file_path=file_url,
            start_offset_ms=startOffsetMs,
            fps=fps,
            duration_ms=None,
        )
        db.add(video)
        db.commit()
        db.refresh(video)
    finally:
        db.close()

    return {
        "status": "ok",
        "videoId": video.id,
        "sessionId": sessionId,
        "filePath": file_url,
        "startOffsetMs": startOffsetMs,
        "fps": fps,
        "durationMs": video.duration_ms,
    }