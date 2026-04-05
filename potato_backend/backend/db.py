from datetime import datetime
import json
from pathlib import Path
import uuid

from sqlalchemy import (Column, Integer, String, Text, DateTime, create_engine, ForeignKey)
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.config import PROJECT_ROOT

# 数据库模型定义
Base = declarative_base()

# 上传记录模型
class UploadRecord(Base):
    __tablename__ = "upload_records"
    id = Column(Integer, primary_key=True)
    original_filename = Column(String(256), nullable=False)
    stored_path = Column(String(512), nullable=False)
    plotted_path = Column(String(512), nullable=True)
    results_json = Column(Text, nullable=True)
    explanation = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# 大模型会话模型
class LLMSession(Base):
    __tablename__ = "llm_sessions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    metadata_json = Column(Text, nullable=True)

# 大模型消息模型
class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("llm_sessions.id"), nullable=False)
    role = Column(String(32), nullable=False)  # user | assistant | system
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# 数据库初始化
DB_DIR = PROJECT_ROOT / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "app.db"

ENGINE = create_engine(f"sqlite:///{DB_PATH.as_posix()}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False)

# 初始化数据库表
def init_db():
    Base.metadata.create_all(bind=ENGINE)

# 保存字节数据到文件
def _save_bytes_to_file(folder: Path, data: bytes, suffix: str):
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex}{suffix}"
    p = folder / name
    with open(p, "wb") as f:
        f.write(data)
    return p

# 保存上传记录
def save_upload_record(original_filename: str, file_bytes: bytes, plotted_bytes: bytes | None, results: list, explanation: str | None):
    """Save uploaded image, plotted image (optional), and create DB record."""
    uploads_dir = PROJECT_ROOT / "uploads"
    stored = _save_bytes_to_file(uploads_dir, file_bytes, Path(original_filename).suffix.lower() or ".jpg")
    plotted_path = None
    if plotted_bytes:
        plotted_path = _save_bytes_to_file(uploads_dir, plotted_bytes, ".jpg")

    session = SessionLocal()
    try:
        rec = UploadRecord(
            original_filename=original_filename,
            stored_path=str(stored.relative_to(PROJECT_ROOT)),
            plotted_path=str(plotted_path.relative_to(PROJECT_ROOT)) if plotted_path else None,
            results_json=json.dumps(results, ensure_ascii=False),
            explanation=explanation,
        )
        session.add(rec)
        session.commit()
        session.refresh(rec)
        return rec.id
    finally:
        session.close()

# 创建大模型会话
def create_llm_session(user_id: int | None = None, metadata: dict | None = None):
    session = SessionLocal()
    try:
        s = LLMSession(user_id=user_id, metadata_json=json.dumps(metadata or {}, ensure_ascii=False))
        session.add(s)
        session.commit()
        session.refresh(s)
        return s.id
    finally:
        session.close()

# 添加消息到大模型会话
def add_message(session_id: int, role: str, content: str):
    session = SessionLocal()
    try:
        m = Message(session_id=session_id, role=role, content=content)
        session.add(m)
        session.commit()
        session.refresh(m)
        return m.id
    finally:
        session.close()

# 查询大模型会话列表
def list_llm_sessions(limit: int = 100, offset: int = 0):
    session = SessionLocal()
    try:
        q = session.query(LLMSession).order_by(LLMSession.started_at.desc()).offset(offset).limit(limit)
        return [
            {
                "id": s.id,
                "user_id": s.user_id,
                "started_at": s.started_at.isoformat(),
                "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                "metadata": json.loads(s.metadata_json) if s.metadata_json else None,
            }
            for s in q
        ]
    finally:
        session.close()

# 查询大模型会话消息列表
def get_messages_by_session(session_id: int):
    session = SessionLocal()
    try:
        q = session.query(Message).filter(Message.session_id == session_id).order_by(Message.created_at.asc())
        return [
            {"id": m.id, "role": m.role, "content": m.content, "created_at": m.created_at.isoformat()}
            for m in q
        ]
    finally:
        session.close()

# 查询上传记录列表
def list_upload_records(limit: int = 100, offset: int = 0):
    session = SessionLocal()
    try:
        q = session.query(UploadRecord).order_by(UploadRecord.created_at.desc()).offset(offset).limit(limit)
        return [
            {
                "id": r.id,
                "original_filename": r.original_filename,
                "stored_path": r.stored_path,
                "plotted_path": r.plotted_path,
                "results": json.loads(r.results_json) if r.results_json else None,
                "explanation": r.explanation,
                "created_at": r.created_at.isoformat(),
            }
            for r in q
        ]
    finally:
        session.close()

# 查询上传记录详情
def get_record_by_id(record_id: int):
    session = SessionLocal()
    try:
        return session.query(UploadRecord).filter(UploadRecord.id == record_id).first()
    finally:
        session.close()
