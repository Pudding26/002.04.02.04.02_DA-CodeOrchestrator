import yaml
from pathlib import Path

def define_DoE_Options(wood_master_df):
    """
    Builds STATIC_FACTORS and DEPENDENT_FACTORS dictionaries dynamically,
    using wood_master_df and YAML-based vocabularies.

    Returns:
        dict: STATIC_FACTORS
        dict: DEPENDENT_FACTORS
    """
    def _load_yaml_keys(yaml_path):
        with open(Path(yaml_path), "r") as f:
            data = yaml.safe_load(f)
        return list(data.keys())

    STATIC_FACTORS = {
        "primary_data.sourceNo": {
            "values": wood_master_df["sourceNo"].unique().tolist(),
            "multi": True
        },
        "primary_data.noShotsRange": {
            "values": [[[2, 15]], [[2, 100]], [[1, 5]], [[10, 25]]],
            "multi": False
        },
        "primary_data.filterNo": {
            "values": _load_yaml_keys("app/config/presets/segmentationFilterPresets.yaml"),
            "multi": False
        },
        "secondary_data.secondaryDataBins": {
            "values": _load_yaml_keys("app/config/presets/secondaryDataBinsPresets.yaml"),
            "multi": False
        },
        "preprocessing.preProcessingNo": {
            "values": _load_yaml_keys("app/config/presets/preProcessingPresets.yaml"),
            "multi": False
        },
        "modeling.metricModelNo": {
            "values": _load_yaml_keys("app/config/presets/metricModelPresets.yaml"),
            "multi": False
        }
    }

    DEPENDENT_FACTORS = {
        "primary_data.view": {
            "multi": True,
            "depends_on": ["primary_data.sourceNo"]
        },
        "primary_data.lens": {
            "multi": True,
            "depends_on": ["primary_data.view", "primary_data.sourceNo"]
        },
        "primary_data.woodType": {
            "multi": True,
            "depends_on": ["primary_data.view", "primary_data.sourceNo"]
        },
        "primary_data.family": {
            "multi": True,
            "depends_on": ["primary_data.view", "primary_data.sourceNo", "primary_data.woodType"]
        },
        "primary_data.genus": {
            "multi": True,
            "depends_on": ["primary_data.view", "primary_data.sourceNo", "primary_data.woodType", "primary_data.family"]
        },
        "primary_data.species": {
            "multi": True,
            "depends_on": ["primary_data.view", "primary_data.sourceNo", "primary_data.woodType", "primary_data.family", "primary_data.genus"]
        }
    }

    return STATIC_FACTORS, DEPENDENT_FACTORS
