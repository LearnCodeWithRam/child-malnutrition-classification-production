"""
SQLAlchemy Data Models for Poshan Malnutrition Screening Stack.
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Text, 
    DateTime, ForeignKey, BigInteger, func
)
from sqlalchemy.orm import relationship
from production.database import Base

class Worker(Base):
    __tablename__ = "workers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    phone_number = Column(String(30), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=True)
    awc_code = Column(String(50), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    def __repr__(self):
        return f"<Worker(phone='{self.phone_number}', awc='{self.awc_code}', active={self.is_active})>"


class ChildAssessment(Base):
    __tablename__ = "child_assessments"

    audit_id = Column(String(64), primary_key=True, index=True)
    worker_phone = Column(String(30), index=True, nullable=False)
    awc_code = Column(String(50), nullable=True)
    serial_number = Column(String(50), nullable=True)
    date_of_birth = Column(String(30), nullable=True)
    gender = Column(String(20), nullable=True)
    height_cm = Column(Float, nullable=True)
    weight_kg = Column(Float, nullable=True)
    muac_cm = Column(Float, nullable=True)
    predicted_tier = Column(String(50), nullable=True)  # e.g., SAM, MAM, Normal, Quality Rejected
    predicted_risk_pct = Column(Float, nullable=True)
    wasting_locus = Column(String(120), nullable=True)
    status = Column(String(30), default="QUEUED", index=True, nullable=False)  # QUEUED, PROCESSING, COMPLETED, REJECTED, FAILED
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    photos = relationship("ChildPhoto", back_populates="assessment", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="assessment", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ChildAssessment(audit_id='{self.audit_id}', status='{self.status}', tier='{self.predicted_tier}')>"


class ChildPhoto(Base):
    __tablename__ = "child_photos"

    photo_id = Column(Integer, primary_key=True, autoincrement=True)
    audit_id = Column(String(64), ForeignKey("child_assessments.audit_id", ondelete="CASCADE"), index=True, nullable=False)
    file_path = Column(String(255), nullable=False)
    sha256_hash = Column(String(64), index=True, nullable=False)
    file_size_bytes = Column(BigInteger, nullable=False)
    mime_type = Column(String(50), default="image/jpeg", nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    assessment = relationship("ChildAssessment", back_populates="photos")

    def __repr__(self):
        return f"<ChildPhoto(audit_id='{self.audit_id}', size={self.file_size_bytes}, hash='{self.sha256_hash[:10]}...')>"


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    audit_id = Column(String(64), ForeignKey("child_assessments.audit_id", ondelete="CASCADE"), index=True, nullable=False)
    execution_time_sec = Column(Float, nullable=True)
    landmark_detected = Column(Boolean, default=False, nullable=False)
    quality_passed = Column(Boolean, default=False, nullable=False)
    pipeline_metrics = Column(Text, nullable=True)  # JSON-serialized diagnostics
    error_log = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    assessment = relationship("ChildAssessment", back_populates="audit_logs")

    def __repr__(self):
        return f"<AuditLog(audit_id='{self.audit_id}', exec_time={self.execution_time_sec}s)>"

