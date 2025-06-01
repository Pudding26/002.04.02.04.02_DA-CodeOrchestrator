from typing import Optional, ClassVar
from app.utils.SQL.models.api_BaseModel import api_BaseModel
from app.utils.SQL.models.temp.orm.DoEJobs import DoEJobs

class DoEJobs_Out(api_BaseModel):
    orm_class: ClassVar = DoEJobs
    db_key: ClassVar[str] = "temp"


    DoE_UUID: str

    ## PrimaryDataFactors##

    sourceNo: str
    woodType: str
    family: str
    genus: str
    species: str
    view: str
    lens: str
    maxShots: str
    noShotsRange: str
    ## SecondaryDataFactors##
    


    ## 2. PreprocessingFactors##
    resampling_method: str
    resampling_strategy: str
    resampling_k_neighbors: str
    resampling_random_state: str

    ## 3. SegmentationFactors##
    preProcessingNo: str

    filterNo: str


    ## 4. ModelFactors##
    featureBins: list
    modelNo: str

