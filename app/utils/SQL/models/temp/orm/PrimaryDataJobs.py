from sqlalchemy import Column, String, Integer, Float, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
from app.utils.SQL.models.orm_BaseModel import orm_BaseModel

class PrimaryDataJobs(orm_BaseModel):
    __tablename__ = "primaryDataJobs"



    sampleID = Column(String, primary_key=True)

    woodType = Column(String)
    species = Column(String)
    family = Column(String)
    genus = Column(String)

    view = Column(String)

    lens = Column(Float)
    totalNumberShots = Column(Integer)
    
    
    #filterNo = Column(Integer)
    DPI = Column(Float)
    pixelSize_um_per_pixel = Column(Float)
    #bitDepth = Column(Integer)
    #colorDepth = Column(String)
    #colorSpace = Column(String)
    #pixel_x = Column(Integer)
    #pixel_y = Column(Integer)
    microscopicTechnic = Column(String)
    area_x_mm = Column(Float)
    area_y_mm = Column(Float)
    numericalAperature_NA = Column(Float)

    IFAW_code = Column(String)
    engName = Column(String)
    deName = Column(String)
    frName = Column(String)
    japName = Column(String)
    samplingPoint = Column(String)
    origin = Column(String)

    citeKey = Column(String)
    institution = Column(String)
    institutionCode = Column(String)
    contributor = Column(String)
    digitizedDate = Column(DateTime)
    sourceNo = Column(String)
    raw_UUID = Column(String, unique=True, nullable=False)

    GPS_Alt = Column(Float)
    GPS_Lat = Column(Float)
    GPS_Long = Column(Float)
    
    sourceFilePath_rel = Column(JSONB)
    hdf5_dataset_path = Column(String)
    #stackID = Column(String) -> not possible on this level
    specimenID = Column(String)
    sourceID = Column(String)