from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from src.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id              = Column(Integer, primary_key=True, index=True)
    input_text      = Column(Text, nullable=False)
    verdict         = Column(String(20), nullable=False)       # Real / Likely Fake / Unverified
    confidence      = Column(Float, nullable=False)
    ml_label        = Column(String(10), nullable=False)       # Real / Fake
    ml_fake_prob    = Column(Float, nullable=False)
    ml_real_prob    = Column(Float, nullable=False)
    news_similarity = Column(Float, nullable=False)
    articles_found  = Column(Integer, nullable=False)
    reason          = Column(Text, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)