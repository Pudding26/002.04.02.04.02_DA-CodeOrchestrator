from __future__ import annotations

from pydantic import BaseModel, ConfigDict, conlist, field_validator
from typing import List, TypeAlias, Literal, Union

from app.utils.dataModels.Jobs.BaseJob import BaseJob


class DoEJob(BaseJob):
    job_type : Literal["general"] = "general"

    doe_config    : DOE_config


    model_config = ConfigDict(populate_by_name=True)


class DOE_config(BaseModel):
    primary_data   : PrimaryData
    segmentation   : SegmentationCfg
    secondary_data : SecondaryData
    preprocessing  : PreProcessingCfg
    modeling       : ModelingCfg


class PrimaryData(BaseModel):
    sourceNo        : List[str]
    woodType        : List[str] 
    family          : List[str]
    genus           : List[str]
    species         : List[str]
    view            : List[str]
    lens            : List[float]
    maxShots        : List[int]
    noShotsRange: List[List[int]]

    model_config = ConfigDict(extra="forbid")

    @field_validator("noShotsRange")
    def check_range_shape(cls, v: Union[List[List[int]], List[None]]) -> Union[List[List[int]], List[None]]:
        if v == [None]:
            return v  # wildcard accepted
        if not isinstance(v, list):
            raise ValueError(f"noShotsRange must be a list, got {type(v)}")
        for pair in v:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"Each noShotsRange entry must be a list of 2 integers, got: {pair}")
            if not all(isinstance(i, int) for i in pair):
                raise ValueError(f"Invalid integers in noShotsRange: {pair}")
        return v



class SegmentationCfg(BaseModel):
    filterNo : List[str]
    model_config = ConfigDict(populate_by_name=True)

class SecondaryData(BaseModel):
    secondaryDataBins : List[str]
    model_config = ConfigDict(populate_by_name=True)

class PreProcessingCfg(BaseModel):
    preProcessingNo : List[str]
    model_config = ConfigDict(populate_by_name=True)

class ModelingCfg(BaseModel):
    metricModelNo : List[str]
    model_config = ConfigDict(populate_by_name=True)