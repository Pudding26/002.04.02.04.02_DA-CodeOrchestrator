import random
import copy
from deap import base, tools, algorithms
from typing import List, Dict, Any
import json
import logging
import os
import warnings
import yaml
from deap import creator
from datetime import datetime
from app.utils.dataModels.Jobs.DoEJob import DoEJob, DEAPMetadata, DEAPInnerMetadata
import time        

import json
import hashlib

import pandas as pd
import json
from app.utils.logger.loggingWrapper import suppress_logging

from app.tasks.TaskBase import TaskBase

from app.tasks.TA28_DoECreator.utils.custom_weighted_selector import custom_weighted_selector
from app.tasks.TA28_DoECreator.utils.get_scores_and_attach import get_scores_and_attach
from app.tasks.TA28_DoECreator.utils.apply_must_rules import apply_must_rules

from app.tasks.TA28_DoECreator.utils.plot_gen_mutation_summary import plot_species_summary
from app.tasks.TA27_DesignOfExperiments.TA27_A_DoEJobGenerator import TA27_A_DoEJobGenerator


LOG_FILE_PATH = "tmp/reports/mutation_stats.csv"
PLOT_FILE_PATH = "tmp/reports/mutation_plots"


class TA28_B_Mutator:
    def __init__(self, last_gen_doe_df, wood_master_df, static_factors, dependent_factors, mutation_config_raw, branch_id, subbranch_id, config_path=None, gen = -1):
        
        self.last_gen_doe_df = last_gen_doe_df
        self.gen = gen
        logging.debug3(f"Initializing TA28_B_Mutator with gen={self.gen}, branch_id={branch_id}, subbranch_id={subbranch_id}, working with len(last_gen_doe_df)={len(last_gen_doe_df)} and len(wood_master_df)={len(wood_master_df)}")
        
        self.branch_id = branch_id
        self.subbranch_id = subbranch_id
        self.shots_per_row = 3.8 # Hardcoded average of shots per stack in WoodMaster, to calculate the potential maxium size of the baseline dataset
        self.theoretical_max_size =  self.shots_per_row * len(wood_master_df)


        self.static_factors = static_factors
        self.dependent_factors = dependent_factors


        self.metadata_dict = self.extract_DEAP_metadata_dict(self.last_gen_doe_df, acc_type = self.branch_id, branch_name = self.subbranch_id)
        self._normalize_metadata_dict()
        self._parse_mutation_config()
        self.parent_job_uuids = self.last_gen_doe_df["job_uuid"].tolist() if "job_uuid" in self.last_gen_doe_df.columns else []

        self.wood_master_df_raw = wood_master_df
        self.wood_master_df = self._apply_must_rules(wood_master_df, self.metadata_dict)
        self.locked_values = self.get_locked_values()



        self.ordered_hierarchy_keys = list(self.static_factors.keys()) + list(self.dependent_factors.keys())




        with open("app/config/DoE/default_mutation_params.yaml", "r") as f:
            defaults = yaml.safe_load(f)

        self.default_mutate_chance = defaults.get("mutate_chance", {})
        self.default_mutation_rate = defaults.get("mutation_rate", {})
        self.default_growth_bias = defaults.get("growth_bias", {})
        self.default_noise_std = defaults.get("noise_std", {})



        self.toolbox = base.Toolbox()
        self._setup_toolbox()
    
    def _setup_toolbox(self):
        self.toolbox.register("select", custom_weighted_selector)
    
        self.toolbox.register("mutate", self.custom_mutate)
        #self.toolbox.register("mate", self.custom_crossover)


    def mutate_generation(self):
        from app.utils.logger.loggingWrapper import LoggingHandler, suppress_logging

        logger = logging.getLogger(__name__)
        LoggingHandler(logging_level=os.getenv("LOGGING_LEVEL", "DEBUG-2"))


        if not hasattr(creator, "FitnessMulti"):
            #creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))  #(score_acc, score_sample, score_entropy)
            creator.create("FitnessSingle", base.Fitness, weights=(1.0,))


        if not hasattr(creator, "Individual"):
            #creator.create("Individual", dict, fitness=creator.FitnessMulti)
            creator.create("Individual", dict, fitness=creator.FitnessSingle)

        raw_df = self.last_gen_doe_df.copy()
        init_pop_with_scores, summary_df = get_scores_and_attach(raw_df, acc_type=self.branch_id, theoretical_max_size=self.theoretical_max_size, subbranch_id=self.subbranch_id, gen=self.gen)
        
        population = self.create_pop(init_pop_with_scores)
        next_gen_size = self.metadata_dict["metadata"]["mutation_config"]["generation_size"]

        selected, selection_summary = self.toolbox.select(population, next_gen_size)
        
        



        concat_df = pd.DataFrame()
        seen_uuids = set()
        max_retries = next_gen_size
        total_retries = 0
        peak_retries = []

        for ind in selected:
            retries = 0

            while retries < max_retries:
                with suppress_logging(logging.ERROR):
                    mutated_ind, did_mutate, mutation_summary = self.toolbox.mutate(ind)
                    
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    single_row_df = self.prepare_next_gen_for_storage([mutated_ind])

                doe_uuid = single_row_df["job_uuid"].iloc[0]

                if concat_df.empty or doe_uuid not in concat_df["job_uuid"].values:
                    concat_df = pd.concat([concat_df, single_row_df], ignore_index=True)
                    break


                else:
                    retries += 1
                    total_retries +=1
                    peak_retries.append(retries)
                    logging.debug1(f"Duplicate job_uuid {doe_uuid} for individual; retrying ({retries}/{max_retries})")

            if retries == max_retries:
                logging.warning(f"Max retries hit for individual; forcibly accepting last attempt.")
                concat_df = pd.concat([concat_df, single_row_df], ignore_index=True)

        #logging.debug3(f"Total retries across all individuals: {total_retries} and a peak of {max(peak_retries)}/{max_retries} avg= {sum(peak_retries)/len(peak_retries):.2f}")
        next_gen_df = concat_df
        summary_df = pd.merge(
            summary_df,
            selection_summary[["DoE_UUID", "front", "weight"]],
            on="DoE_UUID",
            how="inner"
        )
        summary_df.rename(columns={"front": "pareto_front", "weight": "selection_weight"}, inplace=True)
        summary_df["creation_date"] = datetime.now()
        
        keys_to_track = self.ordered_hierarchy_keys.copy()
        keys_to_track.remove("primary_data.noShotsRange")
        keys_to_track.remove("primary_data.filterNo")  # sourceNo is not tracked in mutations

        mutation_stats = compare_generations(
            self.last_gen_doe_df,
            next_gen_df,
            keys_to_track=keys_to_track
        )


        
        LoggingHandler(logging_level=os.getenv("LOGGING_LEVEL", "DEBUG-2"))

        species_df = summarize_species_and_factors(mutation_stats, next_gen_df, 
                                                   branch_id=self.branch_id, 
                                                   subbranch_id=self.subbranch_id, 
                                                   gen=self.gen, 
                                                   len_next_gen_df=len(next_gen_df))




                
        return next_gen_df, summary_df, mutation_summary
        







    def create_pop(self, df):
        """
        Create a DEAP population from a DataFrame.
        
        Each row in the DataFrame must have a `payload` column (JSON or dict) 
        and fitness score columns: `score_acc`, `score_sample`, `score_entropy`.
        If any score is missing, it defaults to a random value in [0, 1].
        
        Returns:
            list: DEAP Individual objects initialized with payload and fitness.
        """

        def init_from_row(row):
            """
            Initialize a DEAP Individual from a single DataFrame row.
            
            Args:
                row (pd.Series): A row from the input DataFrame.

            Returns:
                Individual: DEAP Individual with attached fitness.
            """
            payload = row["payload"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            
            individual = creator.Individual(payload)
            individual.doe_uuid = row["DoE_UUID"]
            
            score_acc = row.get("score_acc", random.uniform(0, 1))
            score_sample = row.get("score_sample", random.uniform(0, 1))
            score_entropy = row.get("score_entropy", random.uniform(0, 1))
            
            #individual.fitness.values = (score_acc, score_sample, score_entropy)
            individual.fitness.values = (score_acc,)
            
            return individual

        return [init_from_row(row) for _, row in df.iterrows()]

    def custom_mutate(self, individual):
        """
        Custom hierarchical mutate method for DEAP individuals.
        Applies configured mutation rates, handles must_contain logic,
        and narrows valid_df progressively down hierarchy.

        Returns:
            (individual, mutated): mutated individual and bool flag.
        """

        mutation_records = []  # Collect summary info per key
        must_mutate = False
        first_trigger_logged = False

        def mutate_list_field(
            current_list,
            valid_options,
            must_contain=None,
            protected=None,
            mutate_chance=0.2,
            mutation_rate=0.3,
            growth_bias=0,
            noise_std=0.1,
            must_mutate=False,
            self = None
        ):
            current = list(current_list or [])
            options = list(valid_options or [])
            must_contain = list(must_contain or [])
            protected = list(protected or [])

            # Compose protected list simply:
            protected_items = dedupe(must_contain + protected)

            # Ensure all current values are valid wrt options
            viable_current = [x for x in current if x in options]



            for item in protected_items:
                if item not in viable_current:
                    viable_current.append(item)


            logging.debug1(
                f"mutate_list_field: current={current}, valid_options={options}, must_contain={must_contain}, protected={protected}"
            )

            # Work on mutation_target = viable_current minus protected_items
            mutation_target = [x for x in viable_current if x not in protected_items]

            missing_overlap = [x for x in current if x not in viable_current]
            overlap_shortfall = len(missing_overlap)

            logging.debug1(f"Overlap shortfall: {overlap_shortfall}")

            if must_mutate == True or viable_current == [] and protected_items == []:
                logging.debug1(f"Must mutate is forced")
                mutate_chance = 1.1  # Force mutation if must_mutate is True

            mutation_roll = random.random()

            if mutation_roll > mutate_chance:
                logging.debug1(f"Mutation skipped (chance={mutate_chance}); returning unchanged")
                return  dedupe(mutation_target + protected_items), 0, 0, 0

            list_size = len(mutation_target)
            noise_factor_drop = random.gauss(1.0, noise_std)
            noise_factor_add = random.gauss(1.0, noise_std)

            eff_rate_drop = max(0, mutation_rate * noise_factor_drop - growth_bias)
            eff_rate_add = max(0, mutation_rate * noise_factor_add + growth_bias)

            drop_n = max(1, int(len(current) * eff_rate_drop))
            add_n = max(1, int(1 + len(current) * eff_rate_add) + overlap_shortfall)

            logging.debug2("Calculated mutation parameters: eff_rate_drop=%.2f, eff_rate_add=%.2f, drop_n=%d, add_n=%d", eff_rate_drop, eff_rate_add, drop_n, add_n)


            drop_sample = random.sample(mutation_target, min(drop_n, len(mutation_target))) if mutation_target else []
            mutated_raw = [x for x in mutation_target if x not in drop_sample]

            addable = [
                x for x in options
                if not any(x == m for m in mutated_raw)
                and not any(x == p for p in protected_items)
            ]
            add_sample = random.sample(addable, min(add_n, len(addable))) if addable else []
            mutated_raw += add_sample

            final = dedupe(mutated_raw + protected_items)


            if len(final) == 0:
                final = current
                logging.warning(f"Final mutated list is empty! Reverting to current: {current}")

            logging.debug1(
                f"mutate_list_field result: drop_n={drop_n}, add_n={add_n}, mutated_raw={mutated_raw}, protected={protected_items}, final={final}"
            )

            return final, drop_n, add_n, overlap_shortfall





        valid_df_inital = self._apply_must_rules(self.wood_master_df, self.metadata_dict)
        initial_valid_df_size = len(valid_df_inital)  # set this when you first obtain valid_df
        valid_df = valid_df_inital.copy()
        logging.debug2(f"---- Starting mutation with initial valid_df size: {initial_valid_df_size} ----")
        for key in self.ordered_hierarchy_keys:
            logging.debug2(f"++++++++++++++++++++ Processing key={key} ++++++++++++++++++++++++++++")

            # Resolve config
            if key in self.static_factors:
                field_config = self.metadata_dict.get("static_mutate_params", {}).get(key, {})
            elif key in self.dependent_factors:
                field_config = self.metadata_dict.get("dependent_mutate_params", {}).get(key, {})
            else:
                field_config = {}
            # Extract mutation parameters
            # Picks a custom value if provided, otherwise falls back to defaults
            mutate_chance = self.custom_mutate_chance.get(key, self.default_mutate_chance.get(key, 0.2))
            mutation_rate = self.custom_mutation_rate.get(key, self.default_mutation_rate.get(key, 0.3))
            growth_bias = self.custom_growth_bias.get(key, self.default_growth_bias.get(key, 0))
            must_contain = self.enriched_must_contain.get(key, [])


            if key == "primary_data.filterNo":
                # Special case for FilterNo: always mutate to a list of valid options
                key_pick = "segmentation.filterNo"
            else:
                key_pick = key




            current_value = self._get_value_from_individual(individual, key_pick)

            # Determine valid options:
            if key in self.static_factors:
                valid_options = self.static_factors[key]["values"]
            else:
                column_name = key.split(".")[-1]
                if column_name in valid_df.columns:
                    valid_options = valid_df[column_name].dropna().unique().tolist()
                else:
                    valid_options = []
                    logging.debug3(f"Column {column_name} not found — fallback empty valid_options.")

            logging.debug2(f"Key={key} current_value={current_value[:3]} (len = {len(current_value)}) | valid_options={valid_options[:3]} (len={len(valid_options)})")
            logging.debug2(f"Mutation params: mutate_chance={mutate_chance}, mutation_rate={mutation_rate}, growth_bias={growth_bias}, must_contain={must_contain}")


            if self.wildcard_flags.get(key, False):
                mutated_value = valid_options.copy()
                logging.debug1(f"Wildcard active for {key}: assigning all valid options {mutated_value}")
            else:
                protected_items = self.locked_values.get(key, [])
                mutated_value, drop_n, add_n, overlap_shortfall = mutate_list_field(
                    current_list=current_value,
                    valid_options=valid_options,
                    must_contain=must_contain,
                    protected=protected_items,
                    mutate_chance=mutate_chance,
                    mutation_rate=mutation_rate,
                    growth_bias=growth_bias,
                    self=self
                )



            if mutated_value != current_value:
                must_mutate = True
                if not first_trigger_logged:
                    logging.debug2(f"[MUTATE] First mutation trigger at key: {key}")
                    first_trigger_logged = True
                logging.debug2(f"Mutation occurred for {key}: {len(current_value)} → {len(mutated_value)}")
                logging.debug1(f"Mutation occurred for {key}: {current_value} → {mutated_value}")
            else:
                logging.debug2(f"No mutation for {key}: len_current_value={len(current_value)} remains unchanged.")

                

            # Record mutation summary
            summary_row = {
                "factor": key,
                "initial_entries": len(current_value),
                "dropped_entries": drop_n,
                "added_entries": add_n,
                "overlap_shortfall": overlap_shortfall,
                "pick_ratio": len(mutated_value) / len(current_value) if current_value else 0,
                "result_entries": len(mutated_value)
            }
            mutation_records.append(summary_row)
            


            if len(mutated_value) == 0:
                logging.critical(f"Mutated value for {key} is empty! Using current value instead.")
                sleep = True
                while sleep:
                    time.sleep(5)


            individual = self._set_value_in_individual(individual, key_pick, mutated_value)

            # Narrow valid_df progressively
            column_name = key.split(".")[-1]
            if column_name in valid_df.columns:
                if mutated_value:
                    preview = mutated_value[:3]
                    last = mutated_value[-1] if len(mutated_value) > 3 else None
                    total = len(mutated_value)

                    preview_str = ", ".join(str(v) for v in preview)
                    if last:
                        preview_str += f", ..., {last}"

                    logging.debug2(f"Filtering valid_df by {column_name}={preview_str} (total={total})")
                rows_before = len(valid_df)
                valid_df = valid_df[valid_df[column_name].isin(mutated_value)]
                rows_after = len(valid_df)

                drop_ratio_iter = (rows_before - rows_after) / rows_before if rows_before else 0
                drop_ratio_total = rows_after / initial_valid_df_size if initial_valid_df_size else 0

                logging.debug2(
                    f"[FILTER] {key}: ({rows_after}/{initial_valid_df_size}); "
                    f"drop_iter={drop_ratio_iter:.1%}; remaining={drop_ratio_total:.1%}"
                )


        # Optionally attach mutation_records as self.last_mutation_summary:
        self.last_mutation_summary = pd.DataFrame(mutation_records)


        return (individual,), must_mutate, mutation_records



    def _get_mutate_chance(self, key):
        cfg = self.mutation_config.get(key, {})
        if "mutate_chance" in cfg:
            return cfg["mutate_chance"]
        return self.default_mutate_chance.get(key, 0.2)

    def _get_mutation_rate(self, key):
        cfg = self.mutation_config.get(key, {})
        if "mutation_rate" in cfg:
            return cfg["mutation_rate"]
        return self.default_mutation_rate.get(key, 0.3)

    def _get_growth_bias(self, key):
        cfg = self.mutation_config.get(key, {})
        if "growth_bias" in cfg:
            return cfg["growth_bias"]
        return self.default_growth_bias.get(key, 0)
    
    def _get_noise_std(self, key):
        cfg = self.mutation_config.get(key, {})
        if "noise_std" in cfg:
            return cfg["noise_std"]
        return self.default_noise_std.get(key, 0.1)

    def _get_value_from_individual(self, individual, dotted_key):
        parts = dotted_key.split(".")
        
        # Assume everything lives under doe_config:
        val = individual.get("doe_config", {})
        
        for part in parts:
            if isinstance(val, dict):
                val = val.get(part)
            else:
                return []
        return val if isinstance(val, list) else []
    
    def _set_value_in_individual(self, individual, dotted_key, value):
        parts = dotted_key.split(".")
        
        # Ensure doe_config root exists:
        val = individual.setdefault("doe_config", {})
        
        for part in parts[:-1]:
            val = val.setdefault(part, {})
        
        val[parts[-1]] = value
        return individual


    
    def extract_DEAP_metadata_dict(self, df, acc_type=None, branch_name=None):
        """
        Parse DEAP_metadata column and extract as dict.
        Optionally filter to a specific acc_type and branch_name.

        Args:
            df (pd.DataFrame): DataFrame with 'DEAP_metadata' column.
            acc_type (str, optional): If provided, extract only this top-level key.
            branch_name (str, optional): If provided, extract only this branch under acc_type.

        Returns:
            dict: Extracted and optionally filtered DEAP_metadata.
        """
        if "DEAP_metadata" not in df.columns:
            raise ValueError("DataFrame must contain 'DEAP_metadata' column.")

        metadata = df.iloc[0]["DEAP_metadata"]

        # Ensure it's parsed:
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON in DEAP_metadata column.")

        # Optional filtering:
        if acc_type and branch_name:
            return metadata.get(acc_type, {}).get(branch_name, {})
        elif acc_type:
            return metadata.get(acc_type, {})
        else:
            return metadata

    def _parse_mutation_config(self):
        """
        Parses self.metadata_dict['metadata']['mutation_config']
        and builds internal mappings:
        - enriched must_contain dict
        - custom mutate_chance dict
        - custom mutation_rate dict
        - custom growth_bias dict
        - wildcard_flags dict
        """
        cfg = self.metadata_dict.get("metadata", {}).get("mutation_config", {})

        enriched_must_contain = {}
        custom_mutate_chance = {}
        custom_mutation_rate = {}
        custom_growth_bias = {}
        wildcard_flags = {}

        section_names = [k for k in cfg.keys() if k in ["static_mutate_params", "dependent_mutate_params"]]
        for section_name in section_names:
            section = cfg.get(section_name, {})
                
            for key, params in section.items():
                # must_contain enrichment and wildcard detection
                must_contain = params.get("must_contain", [])

                is_wildcard = False
                if must_contain is None or must_contain == [None] or must_contain == ["None"]:
                    is_wildcard = True
                else:
                    enriched_must_contain[key] = must_contain

                wildcard_flags[key] = is_wildcard

                # mutate_chance enrichment
                mutate_chance = params.get("mutate_chance")
                if mutate_chance is not None:
                    custom_mutate_chance[key] = mutate_chance

                # mutation_rate enrichment
                mutation_rate = params.get("mutation_rate")
                if mutation_rate is not None:
                    custom_mutation_rate[key] = mutation_rate

                # growth_bias enrichment (optional; fallback 0 if not provided)
                growth_bias = params.get("growth_bias")
                if growth_bias is not None:
                    custom_growth_bias[key] = growth_bias

        # Store these mappings on self
        self.enriched_must_contain = enriched_must_contain
        self.custom_mutate_chance = custom_mutate_chance
        self.custom_mutation_rate = custom_mutation_rate
        self.custom_growth_bias = custom_growth_bias
        self.wildcard_flags = wildcard_flags

        # Optional debug summary
        logging.debug3(f"Wildcard flags: {self.wildcard_flags}")
        logging.debug3(f"Enriched must_contain: {self.enriched_must_contain}")
        logging.debug3(f"Custom mutate_chance: {self.custom_mutate_chance}")
        logging.debug3(f"Custom mutation_rate: {self.custom_mutation_rate}")
        logging.debug3(f"Custom growth_bias: {self.custom_growth_bias}")

    def get_locked_values(self):
        """
        Replicate attach_must_contain logic to determine which fields (woodType, family, genus, species)
        must be locked in based on must_contain config.
        """
        import logging

        locked = {}

        ordered_keys = [
            "primary_data.woodType",
            "primary_data.family",
            "primary_data.genus",
            "primary_data.species"
        ]

        valid_df = self.wood_master_df.copy()
        mutation_cfg = self.metadata_dict.get("metadata", {}).get("mutation_config", {})

        logging.debug3("=== get_locked_values START ===")
        logging.debug3(f"Initial wood_master_df size: {len(valid_df)}")

        for key in ordered_keys:
            section, field = key.split(".")

            params = mutation_cfg.get("static_mutate_params", {}).get(key, {})
            must_contain = params.get("must_contain", [])

            if not must_contain:
                params = mutation_cfg.get("dependent_mutate_params", {}).get(key, {})
                must_contain = params.get("must_contain", [])

            logging.debug3(f"\nProcessing {key}: must_contain={must_contain}")

            if must_contain:
                # Wildcard detection
                is_wildcard = False
                if len(must_contain) == 1:
                    val = must_contain[0]
                    if val is None or (isinstance(val, str) and val.strip().lower() == "none"):
                        is_wildcard = True

                if not is_wildcard:
                    before_size = len(valid_df)
                    valid_df = valid_df[valid_df[field].isin(must_contain)]
                    after_size = len(valid_df)
                    logging.debug3(
                        f"  Applied must_contain={must_contain} on {field} | "
                        f"rows before={before_size}, after={after_size}"
                    )

                    # Lock this key to exactly these must_contain values
                    locked[key] = must_contain
                    logging.debug3(f"  Locked[{key}] = {must_contain}")

                    # Resolve upstream dependencies:
                    if key == "primary_data.species":
                        genus_vals = valid_df["genus"].dropna().unique().tolist()
                        family_vals = valid_df["family"].dropna().unique().tolist()
                        woodtype_vals = valid_df["woodType"].dropna().unique().tolist()
                        logging.debug3(f"    Derived upstream genus={genus_vals}, family={family_vals}, woodType={woodtype_vals}")
                        if genus_vals:
                            locked["primary_data.genus"] = genus_vals
                        if family_vals:
                            locked["primary_data.family"] = family_vals
                        if woodtype_vals:
                            locked["primary_data.woodType"] = woodtype_vals

                    elif key == "primary_data.genus":
                        family_vals = valid_df["family"].dropna().unique().tolist()
                        woodtype_vals = valid_df["woodType"].dropna().unique().tolist()
                        logging.debug3(f"    Derived upstream family={family_vals}, woodType={woodtype_vals}")
                        if family_vals:
                            locked["primary_data.family"] = family_vals
                        if woodtype_vals:
                            locked["primary_data.woodType"] = woodtype_vals

                    elif key == "primary_data.family":
                        woodtype_vals = valid_df["woodType"].dropna().unique().tolist()
                        logging.debug3(f"    Derived upstream woodType={woodtype_vals}")
                        if woodtype_vals:
                            locked["primary_data.woodType"] = woodtype_vals

                else:
                    logging.debug3(f"  Wildcard for {key}, skipping lock")

        logging.debug3(f"\nFinal locked dict: {locked}")
        logging.debug3("=== get_locked_values END ===")

        return locked



    def prepare_next_gen_for_storage(self, mutated_population):
        from app.tasks.TA28_DoECreator.TA28_A_Initiater import TA28_A_Initiater

        next_gen = self.gen + 1
        mutated_records = []

        for individual_tuple in mutated_population:
            # unwrap the DEAP mutate return (tuple)
            individual = individual_tuple[0]

            # flatten config for downstream generator
            flat_row = _flatten_doe_row_mutator(individual["doe_config"])
            flat_row["branch_id"] = self.branch_id
            flat_row["subbranch_id"] = self.subbranch_id

            # ✅ inherit parent gens and append new one
            parent_gens = individual.get("generation", [])
            if isinstance(parent_gens, int):
                parent_gens = [parent_gens]
            flat_row["generation"] = parent_gens + [next_gen]

            # 🔔 Add flat fields that generate() expects:
            flat_row["origin"] = "mutation"
            flat_row["created_by"] = "TA28_DoECreator"
            flat_row["score"] = None
            flat_row["parent_job_uuids"] = self.parent_job_uuids
            flat_row["mutation_config"] = self.metadata_dict.get("metadata", {}).get("mutation_config", {})

            mutated_records.append(flat_row)

        mutated_df = pd.DataFrame(mutated_records)

        with warnings.catch_warnings(record=True):
            warnings.filterwarnings(
                "ignore",
                message="Pydantic serializer warnings",
                category=UserWarning,
                module="pydantic.main"
            )
            with suppress_logging(logging.ERROR):
                jobs = TA27_A_DoEJobGenerator.generate(
                    mutated_df,
                    template_path="app/config/templates/DoE_job_template.yaml"
                )

        job_df = pd.DataFrame([job.to_sql_row() for job in jobs])
        return job_df



    ### UTILS

    def _normalize_metadata_dict(self):
        """
        Safeguard: normalize self.metadata_dict so that mutable config sections
        are always dicts. Prevents downstream breakage if YAML/JSON provided `null`.

        Specifically ensures:
        - metadata['mutation_config'] exists
        - metadata['mutation_config']['static_mutate_params'] is a dict (not None)
        - metadata['mutation_config']['dependent_mutate_params'] is a dict (not None)
        """
        if not isinstance(self.metadata_dict, dict):
            self.metadata_dict = {}

        metadata = self.metadata_dict.setdefault("metadata", {})
        mutation_cfg = metadata.setdefault("mutation_config", {})

        # Normalize potential None values into empty dicts
        for key in ["static_mutate_params", "dependent_mutate_params"]:
            if mutation_cfg.get(key) is None:
                mutation_cfg[key] = {}


    def _apply_must_rules(self, df, mutation_config):
        filtered_df = df.copy()
        all_keys = list(self.static_factors.keys()) + list(self.dependent_factors.keys())

        for key in all_keys:
            section, field = key.split(".")
            params = mutation_config.get("static_mutate_params", {}).get(key, {})
            if not params:
                params = mutation_config.get("dependent_mutate_params", {}).get(key, {})
            if params:
                if field in filtered_df.columns:
                    filtered_df = apply_must_rules(filtered_df, params, field)
                else:
                    logging.debug2(f"Skipping must_contain/must_not_contain for {field} (not in DataFrame)")
        return filtered_df







def _flatten_doe_row_mutator(new_row):
    """
    Converts a nested DoE row (new format) to flat legacy-style format.
    This ensures compatibility with TA27_A_DoEJobGenerator expectations.
    """
    return {
        "woodType": new_row.get("primary_data", {}).get("woodType", []),
        "sourceNo": new_row.get("primary_data", {}).get("sourceNo", []),
        "family": new_row.get("primary_data", {}).get("family", []),
        "genus": new_row.get("primary_data", {}).get("genus", []),
        "species": new_row.get("primary_data", {}).get("species", []),
        "view": new_row.get("primary_data", {}).get("view", []),
        "lens": new_row.get("primary_data", {}).get("lens", []),
        "maxShots": new_row.get("primary_data", {}).get("maxShots", []),
        "noShotsRange": new_row.get("primary_data", {}).get("noShotsRange", []),
        "filterNo": new_row.get("segmentation", {}).get("filterNo", []),  # 🔔 Correct path
        "secondaryDataBins": new_row.get("secondary_data", {}).get("secondaryDataBins", []),
        "preProcessingNo": new_row.get("preprocessing", {}).get("preProcessingNo", []),
        "featureBlockNo": new_row.get("preprocessing", {}).get("featureBlockNo", []),
        "metricModelNo": new_row.get("modeling", {}).get("metricModelNo", []),
    }

def dedupe(seq):
    seen = []
    for item in seq:
        if not any(item == x for x in seen):
            seen.append(item)
    return seen


def compare_generations(old_df: pd.DataFrame, new_df: pd.DataFrame, keys_to_track=None) -> pd.DataFrame:
    """
    Compare payloads between old and new generations.

    Args:
        old_df (pd.DataFrame): DataFrame with column 'payload'
        new_df (pd.DataFrame): DataFrame with column 'payload'
        keys_to_track (list[str], optional): List of dotted keys to compare.
            Defaults to genus/family/species/view/lens/woodType.

    Returns:
        pd.DataFrame: Flattened metrics for all individuals and factors.
    """
    if keys_to_track is None:
        keys_to_track = [
            "primary_data.family",
            "primary_data.genus",
            "primary_data.species",
            "primary_data.view",
            "primary_data.lens",
            "primary_data.woodType",
        ]

    all_metrics = []

    for i, (old_row, new_row) in enumerate(zip(old_df.itertuples(), new_df.itertuples())):
        # Parse payloads
        old_payload = old_row.payload
        if isinstance(old_payload, str):
            old_payload = json.loads(old_payload)
        new_payload = new_row.payload
        if isinstance(new_payload, str):
            new_payload = json.loads(new_payload)

        for dotted_key in keys_to_track:
            parts = dotted_key.split(".")

            old_vals = old_payload.get("doe_config", {})
            new_vals = new_payload.get("doe_config", {})

            for part in parts:
                if isinstance(old_vals, dict):
                    old_vals = old_vals.get(part, {})
                else:
                    old_vals = []
                if isinstance(new_vals, dict):
                    new_vals = new_vals.get(part, {})
                else:
                    new_vals = []

            # Normalize to sets
            old_set = set(old_vals if isinstance(old_vals, list) else [old_vals])
            new_set = set(new_vals if isinstance(new_vals, list) else [new_vals])

            old_count, new_count = len(old_set), len(new_set)
            overlap = len(old_set & new_set)

            metrics = {
                "individual": i,
                "factor": dotted_key,
                "old_count": old_count,
                "new_count": new_count,
                "growth_rate": (new_count - old_count) / max(1, old_count),
                "overlap": overlap,
                "overlap_rate": overlap / max(1, old_count) if old_count else 0,
                "novelty": len(new_set - old_set),
                "novelty_rate": len(new_set - old_set) / max(1, new_count) if new_count else 0,
                "loss_rate": (old_count - overlap) / max(1, old_count) if old_count else 0,
                "jaccard": overlap / len(old_set | new_set) if (old_set | new_set) else 0,
            }
            all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    return metrics_df



def summarize_species_and_factors(mutation_stats_df: pd.DataFrame, new_df: pd.DataFrame, branch_id, subbranch_id, gen, len_next_gen_df):
    import json, logging

    # --- Species stats ---
    species = mutation_stats_df[mutation_stats_df["factor"] == "primary_data.species"].copy()
    rows = []
    if not species.empty:
        species["change"] = species["new_count"] - species["old_count"]

        def make_row(label, col):
            return {
                "section": "species",
                "metric": label,
                "min": species[col].min(),
                "max": species[col].max(),
                "median": species[col].median(),
                "mean": species[col].mean(),
            }

        rows += [
            make_row("historic", "old_count"),
            make_row("actual", "new_count"),
            make_row("change", "change"),
        ]
    species_df = pd.DataFrame(rows)

    # --- Factor counts ---
    categorical_factors = {
        ("primary_data", "sourceNo"): "sourceNo",
        ("secondary_data", "secondaryDataBins"): "secondaryDataBins",
        ("preprocessing", "preProcessingNo"): "preProcessingNo",
        ("modeling", "metricModelNo"): "metricModelNo",
    }

    factor_counts, log_lines = {}, []
    for path_parts, short_name in categorical_factors.items():
        values = []
        for payload in new_df["payload"]:
            if isinstance(payload, str):
                payload = json.loads(payload)
            val = payload.get("doe_config", {})
            for part in path_parts:   # ✅ iterate tuple directly
                if isinstance(val, dict):
                    val = val.get(part, [])
                else:
                    val = []
            if isinstance(val, list):
                values.extend(val)
            elif val is not None:
                values.append(val)

        factor_counts[short_name] = pd.Series(values).value_counts().to_dict()
        log_lines.append(f"{short_name}: {factor_counts[short_name]}")



    log_lines.append("Species stats DataFrame:")
    log_lines.append(f"\n{species_df}")
    logging.debug3("\n".join(log_lines))



    species_df["timestamp"] = datetime.now()
    species_df["branch_id"] = branch_id
    species_df["subbranch_id"] = subbranch_id
    species_df["generation"] = gen
    species_df["generation_size"] = len_next_gen_df




    species_df.to_csv(LOG_FILE_PATH, index=False, mode="a", header=not os.path.exists(LOG_FILE_PATH))

    species_df_total = pd.read_csv(LOG_FILE_PATH)
    if not species_df_total.empty:
        plot_species_summary(df = species_df_total, save_path=LOG_FILE_PATH.replace(".csv", PLOT_FILE_PATH))

    return species_df


