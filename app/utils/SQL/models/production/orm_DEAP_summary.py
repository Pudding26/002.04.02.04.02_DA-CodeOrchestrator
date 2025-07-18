from sqlalchemy import Column, String, Integer, Float, ForeignKey, DateTime, func
from sqlalchemy.orm import relationship
from app.utils.SQL.models.orm_BaseModel import orm_BaseModel

class orm_DEAP_summary(orm_BaseModel):
    __tablename__ = "DEAP_summary"

    id = Column(Integer, primary_key=True, autoincrement=True)
    DoE_UUID = Column(String, nullable=False)
    acc_type = Column(String, nullable=False)
    sub_branch_id = Column(String)  # adjust type if UUID or FK
    gen = Column(Integer)

    score_acc = Column(Float)
    score_sample = Column(Float)
    score_entropy = Column(Float)

    acc_weight_sum = Column(Float)
    sample_weight_sum = Column(Float)
    entropy_weight_sum = Column(Float)

    pareto_front = Column(Integer)
    selection_weight = Column(Float)

    creation_date = Column(DateTime(timezone=True))