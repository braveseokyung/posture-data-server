# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime, timedelta, timezone
import uuid
import boto3
import os
import math

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, DateTime, ForeignKey
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session as OrmSession

# --- S3 설정 (IAM Role 사용) ---
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "posture-video-bucket")
FORWARD_TILT_THRESHOLD_DEG = 25.0  # 거북목 판단 기준 각도 (deg)
PRESIGN_EXPIRES_IN = 3600  # presigned URL 유효 시간(초)
KST = timezone(timedelta(hours=9))

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
    video_file_path = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)  # 세션 시작 시각(UTC)


class SensorSample(Base):
    __tablename__ = "sensor_samples"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(String, index=True, nullable=False)
    device_id = Column(String, index=True, nullable=False)

    offset_ms = Column(Integer, nullable=False)
    ts = Column(DateTime(timezone=True), nullable=False)

    # attitude
    attitude_pitch = Column(Float, nullable=False)
    attitude_roll = Column(Float, nullable=False)
    attitude_yaw = Column(Float, nullable=False)

    # rotation rate
    rot_x = Column(Float, nullable=False)
    rot_y = Column(Float, nullable=False)
    rot_z = Column(Float, nullable=False)

    # gravity
    grav_x = Column(Float, nullable=False)
    grav_y = Column(Float, nullable=False)
    grav_z = Column(Float, nullable=False)

    # user acceleration
    acc_x = Column(Float, nullable=False)
    acc_y = Column(Float, nullable=False)
    acc_z = Column(Float, nullable=False)

class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)

    session_id = Column(String, index=True, nullable=False)
    device_id = Column(String, index=True, nullable=False)

    file_path = Column(String, nullable=False)          # S3 URL
    created_at = Column(DateTime(timezone=True), nullable=False)
    video_start_ts = Column(DateTime(timezone=True), nullable=False) # 세션 시작 시각


# 실제 테이블 생성
Base.metadata.create_all(bind=engine)


# -----------------------------
# Pydantic 스키마
# -----------------------------

class HealthResponse(BaseModel):
    status: str
    timeMs: int

class Attitude(BaseModel):
    pitch: float
    roll: float
    yaw: float


class Vector3(BaseModel):
    x: float
    y: float
    z: float


class SensorSampleIn(BaseModel):
    offsetMs: int
    attitude: Attitude
    rotationRate: Vector3
    gravity: Vector3
    userAcceleration: Vector3


class SensorBatchIn(BaseModel):
    sessionId: str
    deviceId: str
    sessionStartTsMs: int
    samples: List[SensorSampleIn]

class SensorBatchResponse(BaseModel):
    status: str
    sessionId: str
    deviceId: str
    received: int
    firstOffsetMs: int
    lastOffsetMs: int

class VideoPresignRequest(BaseModel):
    sessionId: str
    deviceId: str
    # 확장자, 기본은 iOS mov
    ext: str = "mov"
    sessionStartTsMs: int


class VideoPresignResponse(BaseModel):
    uploadUrl: str  # 여기에 PUT 업로드
    fileKey: str    # S3 object key (나중에 DB 저장용)
    filePath: str   # 공개 URL (분석/조회용)
    expiresIn: int  # presigned URL 유효 시간(초)


class VideoCompleteRequest(BaseModel):
    sessionId: str
    deviceId: str
    fileKey: str               # presign 에서 받은 S3 object key
    sessionStartTsMs: int      # 센서랑 맞춘 세션 시작 epoch ms


class VideoUploadResponse(BaseModel):
    status: str
    videoId: int
    sessionId: str
    deviceId: str
    filePath: str
    sessionStartTsMs: int


class QuaternionPostureIn(BaseModel):
    """
    앱에서 quaternion 하나 보냈을 때 사용하는 입력 스키마
    """
    qw: float
    qx: float
    qy: float
    qz: float


class QuaternionPostureResponse(BaseModel):
    """
    quaternion -> 거북목 판단 결과 반환 스키마
    """
    isTurtleNeck: bool
    pitchDeg: float
    thresholdDeg: float


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
def now_kst() -> datetime:
    return datetime.now(KST)

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

@app.post("/sensor/batch", response_model=SensorBatchResponse)
def upload_sensor_batch(
    batch: SensorBatchIn,
    db: OrmSession = Depends(get_db),
):
    # 1) 샘플 유효성 검사
    if not batch.samples:
        raise HTTPException(status_code=400, detail="No samples provided")

    # 2) 세션 시작 시각
    base_time = datetime.fromtimestamp(batch.sessionStartTsMs / 1000, tz=KST)

    samples_to_add: list[SensorSample] = []

    for s in batch.samples:
        # 세션 시작시간 + offsetMs → 실제 측정 시각
        ts = base_time + timedelta(milliseconds=s.offsetMs)

        samples_to_add.append(
            SensorSample(
                session_id=batch.sessionId,
                device_id=batch.deviceId,
                offset_ms=s.offsetMs,
                ts=ts,
                attitude_pitch=s.attitude.pitch,
                attitude_roll=s.attitude.roll,
                attitude_yaw=s.attitude.yaw,
                rot_x=s.rotationRate.x,
                rot_y=s.rotationRate.y,
                rot_z=s.rotationRate.z,
                grav_x=s.gravity.x,
                grav_y=s.gravity.y,
                grav_z=s.gravity.z,
                acc_x=s.userAcceleration.x,
                acc_y=s.userAcceleration.y,
                acc_z=s.userAcceleration.z,
            )
        )

    db.add_all(samples_to_add)
    db.commit()

    offsets = [s.offsetMs for s in batch.samples]

    return SensorBatchResponse(
        status="ok",
        sessionId=batch.sessionId,
        deviceId=batch.deviceId,
        received=len(batch.samples),
        firstOffsetMs=min(offsets),
        lastOffsetMs=max(offsets),
    )


@app.post("/videos/presign", response_model=VideoPresignResponse)
def get_video_presigned_url(body: VideoPresignRequest):
    # 1) 확장자 정리
    ext = body.ext.lower()
    if ext not in ["mov", "mp4"]:
        raise HTTPException(status_code=400, detail="Unsupported file extension")

    # 2) S3 object key 설계
    base_time = datetime.fromtimestamp(body.sessionStartTsMs / 1000, tz=KST)
    time_str = base_time.strftime("%Y%m%d_%H%M%S")
    file_key = f"videos/{body.sessionId}/{time_str}.{ext}"


    # iOS mov의 Content-Type은 보통 video/quicktime
    content_type = "video/quicktime" if ext == "mov" else "video/mp4"

    # 3) Presigned URL 생성 (클라이언트가 PUT으로 업로드)
    try:
        upload_url = s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": AWS_S3_BUCKET,
                "Key": file_key,
                "ContentType": content_type,
            },
            ExpiresIn=PRESIGN_EXPIRES_IN,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create presigned URL: {e}",
        )

    # 4) S3에 업로드된 후 접근할 공개 URL
    file_url = f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{file_key}"

    return VideoPresignResponse(
        uploadUrl=upload_url,
        fileKey=file_key,
        filePath=file_url,
        expiresIn=PRESIGN_EXPIRES_IN,
    )

@app.post("/videos/complete", response_model=VideoUploadResponse)
def complete_video_upload(
    body: VideoCompleteRequest,
    db: OrmSession = Depends(get_db),
):
    # 1) presign 때와 동일 규칙으로 S3 URL 복원
    file_url = f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{body.fileKey}"

    # 2) epoch ms → datetime(UTC) 변환
    video_start_ts = datetime.fromtimestamp(body.sessionStartTsMs / 1000, tz=KST)

    # 3) DB에 저장
    video = Video(
        session_id=body.sessionId,
        device_id=body.deviceId,
        file_path=file_url,
        video_start_ts=video_start_ts,
        created_at=now_kst(),
    )

    db.add(video)

    # 세션에도 영상 경로 저장
    session = db.query(Session).filter(Session.id == body.sessionId).first()
    if session:
        session.video_file_path = file_url
    
    db.commit()
    db.refresh(video)

    # 4) 응답
    return VideoUploadResponse(
        status="ok",
        videoId=video.id,
        sessionId=body.sessionId,
        deviceId=body.deviceId,
        filePath=file_url,
        sessionStartTsMs=body.sessionStartTsMs,
    )


def quaternion_to_pitch_deg(qw: float, qx: float, qy: float, qz: float) -> float:
    """
    Quaternion(qw, qx, qy, qz) -> pitch(앞/뒤로 숙인 각도, deg)
    - 여기서는 일반적인 yaw-pitch-roll 변환식 중 pitch만 사용
    - 결과는 degree(도) 단위로 반환
    """
    # sin(pitch) 계산
    sinp = 2.0 * (qw * qy - qz * qx)

    # 수치 오차 방지: asin 입력이 -1~1을 넘지 않도록 클램핑
    if abs(sinp) >= 1:
        pitch_rad = math.copysign(math.pi / 2, sinp)  # ±90도
    else:
        pitch_rad = math.asin(sinp)

    pitch_deg = math.degrees(pitch_rad)
    return pitch_deg


def is_turtle_neck_by_pitch(pitch_deg: float) -> bool:
    """
    pitch_deg(머리 숙인 각도, deg)를 기준으로 거북목 여부 판단
    """
    return pitch_deg >= FORWARD_TILT_THRESHOLD_DEG

@app.post("/posture-from-quaternion", response_model=QuaternionPostureResponse)
def posture_from_quaternion(payload: QuaternionPostureIn):
    pitch_deg = quaternion_to_pitch_deg(
        payload.qw, payload.qx, payload.qy, payload.qz
    )
    is_turtle = is_turtle_neck_by_pitch(pitch_deg)

    return QuaternionPostureResponse(
        isTurtleNeck=is_turtle,
        pitchDeg=pitch_deg,
        thresholdDeg=FORWARD_TILT_THRESHOLD_DEG,
    )