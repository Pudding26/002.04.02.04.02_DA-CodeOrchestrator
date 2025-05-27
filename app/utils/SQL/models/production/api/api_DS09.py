# Pydantic Schema + API: app/utils/SQL/models/raw/api/api_DS09Entry.py

from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from sqlalchemy.orm import Session
import logging

from app.utils.SQL.models.api_BaseModel import api_BaseModel

class DS09_Out(api_BaseModel):
    
    id: int
    IFAW_code: Optional[str]
    origin: Optional[str]
    engName: Optional[str]
    deName: Optional[str]
    frName: Optional[str]
    species: Optional[str]
    genus: Optional[str]

    model_config = ConfigDict(from_attributes=True)
