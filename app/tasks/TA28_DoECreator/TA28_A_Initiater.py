"""
TA28_A_Initiater
----------------

This module handles initial population creation for a branch/subbranch
based on constraints and static/dependent factor vocabularies.

It reads from the canonical DEAP_CONFIG template and generates a
DataFrame of valid DoE jobs ready for insertion.
"""

import random
import json
import logging
import warnings
import pandas as pd
from copy import deepcopy
from app.tasks.TA27_DesignOfExperiments.TA27_A_DoEJobGenerator import TA27_A_DoEJobGenerator


from app.utils.logger.loggingWrapper import suppress_logging

class TA28_A_Initiater:

    def __init__(self, wood_master_df, static_factors, dependent_factors):
        self.woodMasterPotential_full = wood_master_df
        self.STATIC_FACTORS = static_factors
        self.DEPENDENT_FACTORS = dependent_factors
        logging.debug3("TA28_A_Initiater initialized with %d static factors and %d dependent factors.",
                      len(static_factors), len(dependent_factors))

    def generate_random_doe_config(self, static_params=None, dependent_params=None, branch_config=None):
        logging.debug3("Starting generate_random_doe_config...")
        config = {
            "primary_data": {},
            "secondary_data": {},
            "preprocessing": {},
            "modeling": {},
            "segmentation": {}
        }

        df = self.woodMasterPotential_full.copy()
        wildcard_flags = self.build_wildcard_flags(branch_config)
        logging.debug2("Wildcard flags: %s", wildcard_flags)

        df = self.filter_out_must_not_contain(df, branch_config)
        logging.debug3("After filter_out_must_not_contain, remaining rows: %d", len(df))

        df = self.extract_static_factors(df, config, static_params)
        logging.debug3("After extract_static_factors, remaining rows: %d", len(df))

        df_must, lowest_field = self.attach_must_contain(df, branch_config, config)
        logging.debug3("After attach_must_contain, lowest_field: %s, rows: %d", lowest_field, len(df_must))

        df = self.extract_dependent_factors(df, config, wildcard_flags)
        logging.debug3("After extract_dependent_factors, remaining rows: %d", len(df))
 

        logging.debug3("Generated config: %s", config)
        config = self.clean_config_dict(config)

        return config

    def filter_out_must_not_contain(self, df, branch_config):
        logging.debug3("Filtering must_not_contain across static and dependent factors.")
        df_result = df.copy()
        for key in list(self.STATIC_FACTORS.keys()) + list(self.DEPENDENT_FACTORS.keys()):
            section, field = key.split(".")
            params = branch_config.get('static_mutate_params', {}).get(key, {})
            must_not_contain = params.get('must_not_contain', [])
            if not must_not_contain:
                params = branch_config.get('dependent_mutate_params', {}).get(key, {})
                must_not_contain = params.get('must_not_contain', [])

            if must_not_contain and field in df_result.columns:
                before = len(df_result)
                df_result = df_result[~df_result[field].isin(must_not_contain)]
                logging.debug2("Filtered %s must_not_contain=%s; rows before=%d after=%d",
                              key, must_not_contain, before, len(df_result))
        return df_result

    def extract_static_factors(self, df, config, static_params, wildcard_flags = {}):
        logging.debug3("Extracting static factors...")
        for key, cfg in self.STATIC_FACTORS.items():
            section, field = key.split(".")
            values = cfg["values"]
            multi = cfg["multi"]

            params = static_params.get(key, {}) if static_params else {}
            must_contain = params.get("must_contain", [])
            must_not_contain = params.get("must_not_contain", [])

            vocab = [v for v in values if v not in must_not_contain]
            if must_contain:
                vocab = [v for v in vocab if v in must_contain]

            logging.debug2("Static factor: %s, filtered vocab: %s", key, vocab)

            if not vocab:
                selected = [] if multi else None
            elif multi:
                if wildcard_flags.get(field, False):
                    selected = vocab
                    logging.debug2("Wildcard match for %s; selected all: %s", field, selected)
                else:
                    selected = random.sample(vocab, k=random.randint(1, len(vocab)))
                    logging.debug2("Random multi sample for %s; selected: %s", field, selected)
            else:
                selected = random.choice(vocab)
                logging.debug2("Random single choice for %s; selected: %s", field, selected)

            config[section][field] = selected
            before = len(df)
            if selected and field in df.columns:
                df = df[df[field].isin(selected if isinstance(selected, list) else [selected])]
                logging.debug2("After narrowing on %s, rows before=%d after=%d", field, before, len(df))
            if selected is []:
                logging.critical(f"Static factor {key} selected empty list! This may lead to no valid DoE jobs.")

        return df

    def attach_must_contain(self, df_filtered, branch_config, config):
        logging.debug3("Attaching must_contain rules (all keys)...")

        ordered_factors = list(self.STATIC_FACTORS.keys()) + list(self.DEPENDENT_FACTORS.keys())

        # -------------------------------------------------------------------------
        # 🔹 Local helper: apply must_not_contain globally on a dataframe
        def _apply_must_not_contain(df):
            for key in list(self.STATIC_FACTORS.keys()) + list(self.DEPENDENT_FACTORS.keys()):
                section, field = key.split(".")
                params = branch_config.get('static_mutate_params', {}).get(key, {})
                must_not_contain = params.get('must_not_contain', [])
                if not must_not_contain:
                    params = branch_config.get('dependent_mutate_params', {}).get(key, {})
                    must_not_contain = params.get('must_not_contain', [])

                if must_not_contain and field in df.columns:
                    before = len(df)
                    df = df[~df[field].isin(must_not_contain)]
                    logging.debug2("Re-cleanup filter %s; rows before=%d after=%d", key, before, len(df))
            return df

        # 🔹 Local helper: get higher hierarchy keys
        def _get_higher_hierarchy_keys(key):
            dependent_keys = list(self.DEPENDENT_FACTORS.keys())
            if key and key in dependent_keys:
                idx = dependent_keys.index(key)
                return [k.split(".")[1] for k in dependent_keys[:idx]]
            return []

        # 🔹 Local helper: get unique non-null values for a field
        def _get_unique_non_null(df, field):
            if field in df.columns:
                return df[field].dropna().unique().tolist()
            return []

        # -------------------------------------------------------------------------
        # Iterate over all keys with must_contain defined
        for key in ordered_factors:
            section, field = key.split(".")
            params = branch_config.get('static_mutate_params', {}).get(key, {})
            must_contain = params.get('must_contain', [])
            if not must_contain:
                params = branch_config.get('dependent_mutate_params', {}).get(key, {})
                must_contain = params.get('must_contain', [])

            if must_contain:
                logging.debug2("Processing must_contain for %s: %s", key, must_contain)

                df_subset = self.woodMasterPotential_full.copy()
                before = len(df_subset)

                # Robust wildcard detection
                is_wildcard = False
                if len(must_contain) == 1:
                    val = must_contain[0]
                    if val is None or (isinstance(val, str) and val.strip().lower() == 'none'):
                        is_wildcard = True

                if is_wildcard:
                    df_subset = df_subset.iloc[0:0]  # Explicitly empty dataframe
                    logging.debug2("Applied must_contain=NaN filter for %s; rows before=%d after=%d",
                                field, before, len(df_subset))
                else:
                    df_subset = df_subset[df_subset[field].isin(must_contain)]
                    logging.debug2("Applied must_contain filter for %s=%s; rows before=%d after=%d",
                                field, must_contain, before, len(df_subset))

                if not is_wildcard:
                    config.setdefault(section, {})[field] = must_contain
                    logging.debug2("Prepopulated field: %s with values: %s", field, must_contain)
                else:
                    logging.debug2("Skipped prepopulating field %s due to wildcard must_contain=%s",
                                field, must_contain)

                # Apply must_not_contain cleanup for this subset
                df_subset = _apply_must_not_contain(df_subset)

                # Prepopulate current field itself
                config.setdefault(section, {})[field] = must_contain
                logging.debug2("Prepopulated field: %s with values: %s", field, must_contain)

                # Determine and prepopulate higher hierarchy fields
                higher_hierarchy_keys = _get_higher_hierarchy_keys(key)
                for higher_field in higher_hierarchy_keys:
                    values = _get_unique_non_null(df_subset, higher_field)
                    config["primary_data"][higher_field] = values
                    logging.debug2("Prepopulated higher hierarchy field %s for %s context: %s",
                                higher_field, field, values)

        # Return original df_filtered unchanged (since this is purely config attachment)
        return df_filtered, None



    def extract_dependent_factors(self, df, config, wildcard_flags={}):
        logging.debug3("Extracting dependent factors...")
        
        df_original = df.copy()
        
        for key in self.DEPENDENT_FACTORS.keys():
            section, field = key.split(".")
            multi = self.DEPENDENT_FACTORS[key]["multi"]

            candidates = df[field].dropna().unique().tolist() if field in df.columns else []
            logging.debug2("Candidates for %s: %s", key, candidates)

            if not candidates:
                selected = [] if multi else None
            elif multi:
                if wildcard_flags.get(field, False):
                    selected = candidates
                    logging.debug2("Wildcard match for %s; selected all", field)
                else:
                    selected = random.sample(candidates, k=random.randint(1, len(candidates)))
                    logging.debug2("Random multi sample for %s; selected: %s", field, selected)
            else:
                selected = random.choice(candidates)
                logging.debug2("Random single choice for %s; selected: %s", field, selected)

            preselected = config.get(section, {}).get(field, [])
            if preselected:
                preselected = preselected if isinstance(preselected, list) else [preselected]
                selected = selected if isinstance(selected, list) else ([selected] if selected is not None else [])
                combined = list(set(preselected) | set(selected))
                config[section][field] = combined
                logging.debug2("Merged preselected and selected for %s: %s", field, combined)
            else:
                config[section][field] = selected

            final_selection = config[section][field]
            before = len(df)
            if final_selection:
                df = df[df[field].isin(final_selection)]
                logging.debug2("After narrowing on dependent %s; rows before=%d after=%d", field, before, len(df))


            if len(df) == 0:
                    logging.critical("No valid rows left after extracting dependent factors! This may lead to no valid DoE jobs.")
                    raise ValueError("No valid rows left after extracting dependent factors!")
            
    
        return df

    def build_wildcard_flags(self, branch_config):
        """
        Build a dictionary mapping field names to boolean flags indicating
        whether they are marked as wildcard based on must_contain = [None].

        Args:
            branch_config (dict): The branch-level configuration including
                                static_mutate_params and dependent_mutate_params.

        Returns:
            dict: { field_name: True/False } where True indicates wildcard behavior.
        """
        wildcard_flags = {}

        for key in list(self.STATIC_FACTORS.keys()) + list(self.DEPENDENT_FACTORS.keys()):
            section, field = key.split(".")

            params = branch_config.get('static_mutate_params', {}).get(key, {})
            must_contain = params.get('must_contain', [])

            if not must_contain:
                params = branch_config.get('dependent_mutate_params', {}).get(key, {})
                must_contain = params.get('must_contain', [])

            if must_contain and len(must_contain) == 1:
                val = must_contain[0]
                if val is None or (isinstance(val, str) and val.lower() == 'none'):
                    wildcard_flags[field] = True
                else:
                    wildcard_flags[field] = False
            else:
                wildcard_flags[field] = False

        return wildcard_flags
    

    def clean_config_dict(self, d):
        """
        Recursively clean a config dict:
        - For any list, remove None and 'None' (case-insensitive)
        - Preserve dict/list structure
        """
        if isinstance(d, dict):
            return {k: self.clean_config_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [
                self.clean_config_dict(v) for v in d
                if v is not None and not (isinstance(v, str) and v.strip().lower() == 'none')
            ]
        else:
            return d


    def create_initial_population(self, branch_id: str, subbranch_id: str, branch_config: dict) -> pd.DataFrame:
        """
        Create initial generation for a given branch/subbranch.

        Args:
            branch_id (str): Branch name
            subbranch_id (str): Subbranch name
            branch_config (dict): Config from DEAP_CONFIG YAML for this branch.

        Returns:
            pd.DataFrame: Rows ready to insert into DoEJobs table.
        """
        generation_size = branch_config.get("generation_size", 30)
        static_params = branch_config.get("static_mutate_params", {})
        dependent_params = branch_config.get("dependent_mutate_params", {})

        flat_rows = []

        for _ in range(generation_size):

            with suppress_logging():
                base_cfg = self.generate_random_doe_config(
                    static_params=static_params,
                    dependent_params=dependent_params,
                    branch_config=branch_config
                )


            flat_cfg = self._flatten_doe_row(base_cfg)

            flat_cfg.update({
                "branch_id": branch_id,
                "subbranch_id": subbranch_id,
                "generation": 0,
                "origin": "init",
                "created_by": "TA28_DoECreator",
                "mutation_config": {
                    "mutation_strategy": branch_config.get("mutation_strategy"),
                    "mutation_func": branch_config.get("mutation_func"),
                    "allow_multi_mutation": branch_config.get("allow_multi_mutation"),
                    "fallback_on_invalid": branch_config.get("fallback_on_invalid"),
                    "max_generations": branch_config.get("max_generations"),
                    "generation_size": branch_config.get("generation_size"),
                    "elite_size": branch_config.get("elite_size"),
                    "static_mutate_params": branch_config.get("static_mutate_params"),
                    "dependent_mutate_params": branch_config.get("dependent_mutate_params"),
                }
            })


            flat_rows.append(flat_cfg)



        df = pd.DataFrame(flat_rows)

        jobs = TA27_A_DoEJobGenerator.generate(
            df=df, template_path="app/config/templates/DoE_job_template.yaml"
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            job_df = pd.DataFrame([job.to_sql_row() for job in jobs])

        return job_df

    @staticmethod
    def _flatten_doe_row(new_row):
        """
        Converts a nested DoE row (new format) to flat legacy-style format.
        """
        return {
            "woodType": new_row["primary_data"].get("woodType", ["None"]),
            "sourceNo": new_row["primary_data"].get("sourceNo", ["None"]),
            "family": new_row["primary_data"].get("family", ["None"]),
            "genus": new_row["primary_data"].get("genus", ["None"]),
            "species": new_row["primary_data"].get("species", ["None"]),
            "view": new_row["primary_data"].get("view", ["None"]),
            "lens": new_row["primary_data"].get("lens", ["None"]),
            "filterNo": new_row["primary_data"].get("filterNo", ["None"]),
            "secondaryDataBins": new_row["secondary_data"].get("secondaryDataBins", ["None"]),
            "preProcessingNo": new_row["preprocessing"].get("preProcessingNo", ["None"]),
            "featureBlockNo": new_row["preprocessing"].get("featureBlockNo", ["None"]),
            "metricModelNo": new_row["modeling"].get("metricModelNo", ["None"]),
            "maxShots": new_row["primary_data"].get("maxShots", [500]),
            "noShotsRange": new_row["primary_data"].get("noShotsRange", [30])
        }


# 🔹 Debugging check for empty lists in config:
def _check_for_empty_lists(d, path=""):
    check = False
    for k, v in d.items():
        current_path = f"{path}.{k}" if path else k
        if isinstance(v, dict):
            _check_for_empty_lists(v, current_path)
        elif isinstance(v, list) and len(v) == 0:
            check = True
            return check
            # Optional: raise exception for hard fail during debug:
            # raise ValueError(f"Empty list at config path: {current_path}")

