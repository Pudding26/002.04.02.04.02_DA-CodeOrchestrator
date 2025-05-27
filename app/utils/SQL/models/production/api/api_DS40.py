# Pydantic Schema + API: app/utils/SQL/models/raw/api/api_DS09Entry.py

from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from sqlalchemy.orm import Session
import logging
from app.utils.SQL.models.api_BaseModel import api_BaseModel


class DS40_Out(api_BaseModel):
    
    genus: int
    family: str


