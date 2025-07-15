from __future__ import annotations
from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, ConfigDict

# ✅ Import job infrastructure
from app.utils.SQL.models.jobs.orm_WorkerJobs import orm_WorkerJobs
from app.utils.dataModels.Jobs.BaseJob import BaseJob
from app.utils.dataModels.Jobs.JobEnums import JobStatus, JobKind

# ✅ Import configuration sub-models
from app.utils.dataModels.configs.scaling import ScalingConfig
from app.utils.dataModels.configs.metricModelling import DimReductionCfg, BinningCfg


# ─────────────────────────────────────────────────────────────────────────────
# STEP STATUS ENUM
# ─────────────────────────────────────────────────────────────────────────────

StepStatus = Union[
    Literal["passed"],  # Normal success
    str                 # Failure or explanation reason
]

# ─────────────────────────────────────────────────────────────────────────────
# FAIL TRAIL STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

class ModelerFailTrail(BaseModel):
    """
    Stores detailed pass/fail outcomes for each pipeline stage.
    Enables root-cause tracking in logs, UI, and reports.
    """
    preprocessing: Dict[str, StepStatus] = Field(default_factory=dict)
    modelling: Dict[str, StepStatus] = Field(default_factory=dict)
    validation: Dict[str, StepStatus] = Field(default_factory=dict)

    def mark(self, stage: Literal["preprocessing", "modelling", "validation"], step: str, result: StepStatus):
        """Mark the result of a specific step in a stage."""
        getattr(self, stage)[step] = result

# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAPPING CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class BootstrappingConfig(BaseModel):
    """
    Configuration for running N bootstrap resampling iterations on the input data.
    """
    enabled: bool = False                # Activate bootstrapping
    n_iterations: int = 10              # How many bootstrap replicates to run
    n_samples: Optional[int] = None     # Optional override for sample size (defaults to len(data))


# ─────────────────────────────────────────────────────────────────────────────
# RESAMPLING CONFIGURATION (SIMPLIFIED)
# ─────────────────────────────────────────────────────────────────────────────

class RandomSamplerConfig(BaseModel):
    scope: str
    sampling_strategy: str
    random_state: Optional[int] = 42
    mode: Optional[Literal["under", "over", "hybrid"]] = "over"
    strategy: Optional[Literal["mean", "median", "min", "max"]] = "median"

class SMOTESamplerConfig(BaseModel):
    scope: str
    k_neighbors: int = 5
    sampling_strategy: str
    random_state: Optional[int] = 42

class ResamplingConfig(BaseModel):
    method: Literal["RandomSampler", "SMOTESampler"]
    RandomSampler: Optional[RandomSamplerConfig] = None
    SMOTESampler: Optional[SMOTESamplerConfig] = None

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class PreProcessingAttributes(BaseModel):
    """
    Configuration block controlling data preprocessing steps before modeling.
    """
    scaling: Optional[ScalingConfig] = Field(default_factory=ScalingConfig)
    resampling: Optional[ResamplingConfig] = None
    bootstrapping: Optional[BootstrappingConfig] = Field(default_factory=BootstrappingConfig)

# ─────────────────────────────────────────────────────────────────────────────
# METRIC MODELING CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class MetricModelAttributes(BaseModel):
    """
    Configuration block controlling feature binning and dimensionality reduction.
    """
    binning_cfg: Optional[BinningCfg] = Field(default_factory=BinningCfg)
    dim_reduction_cfg: Optional[DimReductionCfg] = Field(default_factory=DimReductionCfg)

# ─────────────────────────────────────────────────────────────────────────────
# JOB INPUT (INSTRUCTIONS)
# ─────────────────────────────────────────────────────────────────────────────

class ModelerJobInput(BaseModel):
    """
    Contains all pipeline instructions (preprocessing + modeling), 
    and basic job-level identifiers.
    """
    stackIDs: List[str]                                 # Required input samples

    preProcessing_instructions: PreProcessingAttributes = Field(
        default_factory=PreProcessingAttributes,
        description="Preprocessing config (scaling, resampling, bootstrapping)"
    )
    metricModel_instructions: MetricModelAttributes = Field(
        default_factory=MetricModelAttributes,
        description="Model configuration (binning, dimensionality reduction)"
    )

    job_No: Optional[int] = None
    preProcessingNo: str                                # Preset ID
    metricModelNo: str                                  # Preset ID

    scope: Optional[str] = None                         # Taxonomic scope for resampling (e.g., species)
    index_col: Optional[int] = None                     # Encoded label column (for classification)

    bootstrap_iteration: int = 0                        # 0 = normal run; 1-N = bootstrapped replicate
    fail_trail: Optional[ModelerFailTrail] = Field(default_factory=ModelerFailTrail)

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ─────────────────────────────────────────────────────────────────────────────
# JOB ATTRIBUTES (STATE + RESULTS)
# ─────────────────────────────────────────────────────────────────────────────

class ModelerAttrs(BaseModel):
    """
    Contains the state of the job during/after execution:
    raw inputs, transformed matrices, modeling output, validation results.
    """
    raw_data: Optional[Any] = None                      # Raw pandas.DataFrame before processing
    preProcessed_data: Optional[Any] = None             # Preprocessed array (cupy/numpy)
    data_num: Optional[Any] = None                      # Numeric features ready for modeling
    model_results: Optional[Any] = None                 # Pandas DF from model stage

    encoder: Optional[Any] = None                       # Column name → index encoder dict
    colname_encoder: Optional[List[str]] = None         # Index → column name list

    engineered_data: Optional[Any] = None               # Output of feature engineering step
    multi_pca_results: Optional[Any] = None             # Dict[frac → PCA result]
    blacklist: Optional[List[str]] = None               # Feature names excluded from modeling
    dropped_fraction: Optional[float] = None            # % of rows dropped due to quality filters

    featureClusterMap: Optional[Any] = None             # UMAP / cluster visualization data
    result_df: Optional[Any] = None                     # Final metrics table
    results_cupy: Optional[Any] = None                  # Intermediate scores (GPU-based)

    uniques: Optional[Dict[str, int]] = None            # Class counts for labels
    bin_dict: Optional[Dict[str, Any]] = None            # Binning results (if binning applied)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN JOB CLASS
# ─────────────────────────────────────────────────────────────────────────────

class ModelerJob(BaseJob):
    """
    Top-level modeling job object passed through the pipeline. 
    Orchestrator executes the job using its `.input` and updates `.attrs` and `.stats`.
    """
    job_type: JobKind = JobKind.MODELER
    orm_model = orm_WorkerJobs
    status: JobStatus = JobStatus.READY.value

    input: ModelerJobInput                              # Pipeline config + scope
    attrs: ModelerAttrs                                 # Data + results
    stats: Dict[str, Any] = Field(default_factory=dict) # Time, dimensions, accuracy, etc.
    context: Optional[Dict[str, Any]] = None            # Summary of pipeline steps (for UI/logs)

    model_config = ConfigDict(extra="forbid")
