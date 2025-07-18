from typing import Optional, ClassVar
from datetime import datetime
from app.utils.SQL.models.api_BaseModel import api_BaseModel

from app.utils.SQL.models.production.orm_DEAP_summary import orm_DEAP_summary

class DEAP_summary_Out(api_BaseModel):

    orm_class: ClassVar = orm_DEAP_summary
    db_key: ClassVar[str] = "production"

    DoE_UUID: str
    acc_type: str
    sub_branch_id: Optional[str] = None
    gen: Optional[int] = None

    score_acc: Optional[float] = None
    score_sample: Optional[float] = None
    score_entropy: Optional[float] = None

    acc_weight_sum: Optional[float] = None
    sample_weight_sum: Optional[float] = None
    entropy_weight_sum: Optional[float] = None

    pareto_front: Optional[int] = None
    selection_weight: Optional[float] = None

    creation_date: Optional[datetime] = None