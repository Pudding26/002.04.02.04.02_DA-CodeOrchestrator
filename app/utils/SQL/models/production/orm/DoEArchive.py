from app.utils.SQL.models.orm_BaseModel import orm_BaseModel



from sqlalchemy import Column, String
from app.utils.SQL.models.orm_BaseModel import orm_BaseModel

class DoEArchive(orm_BaseModel):
    __tablename__ = "DoEArchive"

    DoE_UUID = Column(String, primary_key=True)
    
    
    ## PrimaryDataFactors##
    sourceNo = Column(String)
    woodType = Column(String)
    family = Column(String)
    genus = Column(String)
    species = Column(String)
    
    view = Column(String)
    lens = Column(String)
    
    maxShots = Column(String)
    
    noShotsRange = Column(String)
    
    ## SecondaryDataFactors##


    ## 2. PreprocessingFactors##
    preProcessingNo = Column(String)


    ## 3. SegmentationFactors##
    filterNo = Column(String)

    
    ## 4. ModelFactors##
    featureBins = Column(List)
    modelNo = Column(String)