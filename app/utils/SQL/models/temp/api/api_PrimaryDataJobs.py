from typing import Optional, List, ClassVar
from datetime import datetime
from app.utils.SQL.models.api_BaseModel import api_BaseModel


from app.utils.SQL.models.temp.orm.PrimaryDataJobs import PrimaryDataJobs

class PrimaryDataJobs_Out(api_BaseModel):


    orm_class: ClassVar = PrimaryDataJobs
    db_key: ClassVar[str] = "temp"

    sampleID: str

    woodType: Optional[str]
    species: Optional[str]
    family: Optional[str]
    genus: Optional[str]

    view: Optional[str]

    lens: Optional[float]
    totalNumberShots: Optional[int]

    #filterNo: Optional[int]
    DPI: Optional[int]
    pixelSize_um_per_pixel: Optional[float]
    #bitDepth: Optional[int]
    #colorDepth: Optional[str]
    #colorSpace: Optional[str]
    #pixel_x: Optional[int]
    #pixel_y: Optional[int]
    microscopicTechnic: Optional[str]
    area_x_mm: Optional[float]
    area_y_mm: Optional[float]
    numericalAperature_NA: Optional[float]

    IFAW_code: Optional[str]
    engName: Optional[str]
    deName: Optional[str]
    frName: Optional[str]
    japName: Optional[str]
    samplingPoint: Optional[str]
    origin: Optional[str]

    citeKey: Optional[str]
    institution: Optional[str]
    institutionCode: Optional[str]
    contributor: Optional[str]
    digitizedDate: Optional[datetime]
    sourceNo: Optional[str]
    raw_UUID: str

    GPS_Alt: Optional[float]
    GPS_Lat: Optional[float]
    GPS_Long: Optional[float]

    sourceFilePath_rel: Optional[str]
    hdf5_dataset_path: Optional[str]
    stackID: Optional[str]
    specimenID: Optional[str]
    sourceID: Optional[str]


