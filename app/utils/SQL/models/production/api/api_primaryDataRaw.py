# Pydantic Schema + API: app/utils/SQL/models/raw/api/api_PrimaryDataRaw.py

from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from sqlalchemy.orm import Session
import logging

from app.utils.SQL.DBEngine import DBEngine
from app.utils.SQL.models.production.orm.PrimaryDataRaw import PrimaryDataRaw

class PrimaryDataRawOut(BaseModel):

    UUID: int
    citeKey: Optional[str]
    filename: Optional[str]
    sourceNo: Optional[str]
    path: Optional[str]
    sourceFilePath_rel: Optional[str]
    species: str
    genus: Optional[str]
    source_UUID: Optional[str]
    filename_drop: Optional[str]
    max_split_drop: Optional[str]
    specimenID_old: Optional[str]
    shotNo: Optional[str]
    specimenNo: Optional[str]
    pixel_x: Optional[int]
    pixel_y: Optional[int]
    family: Optional[str]
    GPS_Alt: Optional[float]
    GPS_Lat: Optional[float]
    GPS_Long: Optional[float]
    japName: Optional[str]
    specimenNo_old: Optional[str]
    order_old: Optional[str]
    section_drop: Optional[str]
    subgenus_drop: Optional[str]
    anatomy1_DS_4: Optional[str]
    anatomy2_DS_4: Optional[str]
    otherNo_drop: Optional[str]
    prepNo_drop: Optional[str]
    samplingPoint: Optional[str]
    subspecies_drop: Optional[str]
    individuals_drop: Optional[str]
    n_individuals_drop: Optional[str]
    view: Optional[str]
    lens: Optional[int]
    microscopicTechnic: Optional[str]
    bitDepth: Optional[str]
    colorSpace: Optional[str]
    colorDepth: Optional[str]
    DPI: Optional[str]
    totalNumberShots: Optional[int]
    institution: Optional[str]
    contributor: Optional[str]
    origin: Optional[str]
    digitizedDate: Optional[str]
    area_x_mm: Optional[float]
    area_y_mm: Optional[float]
    pixelSize_um_per_pixel: Optional[float]
    numericalAperature_NA: Optional[float]
    sourceStoredLocally: Optional[bool]
    institutionCode: Optional[str]
    sourceFilePath_abs: Optional[str]
    woodType: Optional[str]
    name_drop: Optional[str]
    todo_lens: Optional[str]
    filename_old: Optional[str]
    source_drop: Optional[str]
    todo_bitdepth_old: Optional[str]
    engName_old: Optional[str]
    version_old: Optional[str]
    contributor_old: Optional[str]
    institution_old: Optional[str]
    view_old: Optional[str]

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def fetch_all(cls, db_key="raw") -> List["PrimaryDataRawOut"]:
        session: Session = DBEngine(db_key).get_session()
        try:
            results = session.query(PrimaryDataRaw).all()
            return [cls.model_validate(r) for r in results]
        except Exception as e:
            logging.error(f"❌ Failed to fetch primaryDataRaw entries: {e}", exc_info=True)
            return []
        finally:
            session.close()
