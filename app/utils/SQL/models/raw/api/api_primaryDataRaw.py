from pydantic import BaseModel, ConfigDict
from typing import Optional, List, ClassVar, Any


from app.utils.SQL.models.api_BaseModel import api_BaseModel

from app.utils.SQL.models.raw.orm.PrimaryDataRaw import PrimaryDataRaw

class PrimaryDataRawOut(api_BaseModel):

    orm_class: ClassVar = PrimaryDataRaw


    raw_UUID: str
    citeKey: Optional[Any]
    filename: Optional[Any]
    sourceNo: Optional[Any]
    path: Optional[Any]
    sourceFilePath_rel: Optional[Any]
    species: Any
    genus: Optional[Any]
    source_UUID: Optional[Any]
    filename_drop: Optional[Any]
    max_split_drop: Optional[Any]
    specimenID_old: Optional[Any]
    shotNo: Optional[Any]
    specimenNo: Optional[Any]
    pixel_x: Optional[Any]
    pixel_y: Optional[Any]
    family: Optional[Any]
    GPS_Alt: Optional[Any]
    GPS_Lat: Optional[Any]
    GPS_Long: Optional[Any]
    japName: Optional[Any]
    specimenNo_old: Optional[Any]
    order_old: Optional[Any]
    section_drop: Optional[Any]
    subgenus_drop: Optional[Any]
    otherNo_drop: Optional[Any]
    prepNo_drop: Optional[Any]
    samplingPoint: Optional[Any]
    subspecies_drop: Optional[Any]
    individuals_drop: Optional[Any]
    n_individuals_drop: Optional[Any]
    view: Optional[Any]
    lens: Optional[Any]
    microscopicTechnic: Optional[Any]
    bitDepth: Optional[Any]
    colorSpace: Optional[Any]
    colorDepth: Optional[Any]
    DPI: Optional[Any]
    totalNumberShots: Optional[Any]
    institution: Optional[Any]
    contributor: Optional[Any]
    origin: Optional[Any]
    digitizedDate: Optional[Any]
    area_x_mm: Optional[Any]
    area_y_mm: Optional[Any]
    pixelSize_um_per_pixel: Optional[Any]
    numericalAperature_NA: Optional[Any]
    sourceStoredLocally: Optional[Any]
    institutionCode: Optional[Any]
    sourceFilePath_abs: Optional[Any]
    woodType: Optional[Any]
    name_drop: Optional[Any]
    todo_lens: Optional[Any]
    filename_old: Optional[Any]
    source_drop: Optional[Any]
    todo_bitdepth_old: Optional[Any]
    engName_old: Optional[Any]
    contributor_old: Optional[Any]
    institution_old: Optional[Any]
    view_old: Optional[Any]
    anatomy1_DS04: Optional[Any]
    anatomy2_DS04: Optional[Any]
    version_old: Optional[Any]


