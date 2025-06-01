from typing import Optional, ClassVar
from app.utils.SQL.models.api_BaseModel import api_BaseModel
from app.utils.SQL.models.production.orm.DoEArchive import DoEArchive

class DoEArchive_Out(api_BaseModel):
    orm_class: ClassVar = DoEArchive
    db_key: ClassVar[str] = "production"
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

