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
from app.utils.dataModels.Jobs.DoEJob import DoEJob, DEAPMetadata, DEAPInnerMetadata

import pandas as pd
import json




class TA28_B_Mutator:
    def __init__(self, last_gen_doe_df, wood_master_df, static_factors, dependent_factors, mutation_config_raw, branch_id, subbranch_id, config_path=None):
        
        self.last_gen_doe_df = last_gen_doe_df
        row = last_gen_doe_df.iloc[0]
        
        self.branch_id = branch_id
        self.subbranch_id = subbranch_id


        self.metadata_dict = self.extract_DEAP_metadata_dict(self.last_gen_doe_df, acc_type = self.branch_id, branch_name = self.subbranch_id)
        self._parse_mutation_config()


        self.wood_master_df = wood_master_df
        self.static_factors = static_factors
        self.dependent_factors = dependent_factors
        self.ordered_hierarchy_keys = list(self.static_factors.keys()) + list(self.dependent_factors.keys())
        self.gen_config = mutation_config_raw


        self.mutation_config = self._get_global_mutation_config()

        with open("app/config/DoE/default_mutation_params.yaml", "r") as f:
            defaults = yaml.safe_load(f)

        self.default_mutate_chance = defaults.get("mutate_chance", {})
        self.default_mutation_rate = defaults.get("mutation_rate", {})
        self.default_growth_bias = defaults.get("growth_bias", {})
        self.default_noise_std = defaults.get("noise_std", {})



        self.toolbox = base.Toolbox()
        self._setup_toolbox()
    
    def _setup_toolbox(self):
        self.toolbox.register("select", tools.selNSGA2)
        self.toolbox.register("mutate", self.custom_mutate)
        self.toolbox.register("mate", self.custom_crossover)


    def mutate_generation(self):
        
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))  # Example weights for 3 objectives

        if not hasattr(creator, "Individual"):
            creator.create("Individual", dict, fitness=creator.FitnessMulti)
        raw_df = self.last_gen_doe_df.copy()
        init_pop_with_scores = self.get_scores_and_attach(raw_df)
        
        population = self.create_pop(init_pop_with_scores)
        next_gen_size = len(population)

        selected = self.toolbox.select(population, next_gen_size)
        ind = selected[0]  # Select the first individual for mutation
        mutated_ind, did_mutate = self.toolbox.mutate(ind)
        
        
        



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
            
            score_acc = row.get("score_acc", random.uniform(0, 1))
            score_sample = row.get("score_sample", random.uniform(0, 1))
            score_entropy = row.get("score_entropy", random.uniform(0, 1))
            
            individual.fitness.values = (score_acc, score_sample, score_entropy)
            
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
        mutated = False

        def mutate_list_field(
            current_list,
            valid_options,
            must_contain=None,
            mutate_chance=0.2,
            mutation_rate=0.3,
            growth_bias=0,
            noise_std=0.1,
        ):
            current = list(current_list or [])
            options = list(valid_options or [])
            must_contain = list(must_contain or [])

            logging.debug2(f"mutate_list_field: current={current}, valid_options={options}, must_contain={must_contain}")

            if random.random() > mutate_chance:
                logging.debug2(f"mutate_list_field: mutation skipped (chance={mutate_chance}) for current={current}")
                return current

            initial_set = set(current)
            initial_set.update(must_contain)

            list_size = len(initial_set)
            noise_factor_drop = random.gauss(1.0, noise_std)
            noise_factor_add = random.gauss(1.0, noise_std)

            eff_rate_drop = max(0, mutation_rate * noise_factor_drop)
            eff_rate_add = max(0, mutation_rate * noise_factor_add)

            drop_n = int(list_size * eff_rate_drop)
            add_n = int(list_size * eff_rate_add + growth_bias)

            droppable = [item for item in initial_set if item not in must_contain]
            drop_sample = random.sample(droppable, min(drop_n, len(droppable))) if droppable else []

            mutated = [item for item in initial_set if item not in drop_sample]

            addable = [item for item in options if item not in mutated]
            add_sample = random.sample(addable, min(add_n, len(addable))) if addable else []
            mutated.extend(add_sample)

            for item in must_contain:
                if item not in mutated:
                    mutated.append(item)

            logging.debug2(f"mutate_list_field: drop_n={drop_n}, add_n={add_n}, final={mutated}")
            return mutated

        valid_df = self._apply_must_not_contain(self.wood_master_df, self.mutation_config)

        for key in self.ordered_hierarchy_keys:
            logging.debug3(f"Processing key={key}")

            # Resolve config
            if key in self.static_factors:
                field_config = self.mutation_config.get("static_mutate_params", {}).get(key, {})
            elif key in self.dependent_factors:
                field_config = self.mutation_config.get("dependent_mutate_params", {}).get(key, {})
            else:
                field_config = {}
            # Extract mutation parameters
            # Picks a custom value if provided, otherwise falls back to defaults
            mutate_chance = self.custom_mutate_chance.get(key, self.default_mutate_chance.get(key, 0.2))
            mutation_rate = self.custom_mutation_rate.get(key, self.default_mutation_rate.get(key, 0.3))
            growth_bias = self.custom_growth_bias.get(key, self.default_growth_bias.get(key, 0))
            must_contain = self.enriched_must_contain.get(key, [])




            current_value = self._get_value_from_individual(individual, key)

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

            logging.debug3(f"Key={key} current_value={current_value} valid_options={valid_options}")
            logging.debug3(f"Mutation params: mutate_chance={mutate_chance}, mutation_rate={mutation_rate}, growth_bias={growth_bias}, must_contain={must_contain}")


            if self.wildcard_flags.get(key, False):
                mutated_value = valid_options.copy()
                logging.debug2(f"Wildcard active for {key}: assigning all valid options {mutated_value}")
            else:

                mutated_value = mutate_list_field(
                    current_list=current_value,
                    valid_options=valid_options,
                    must_contain=must_contain,
                    mutate_chance=mutate_chance,
                    mutation_rate=mutation_rate,
                    growth_bias=growth_bias,
                )

            if mutated_value != current_value:
                mutated = True
                logging.debug3(f"Mutation occurred for {key}: {current_value} → {mutated_value}")

            # Record mutation summary
            summary_row = {
                "factor": key,
                "initial_entries": len(current_value),
                "dropped_entries": len(current_value) - len(mutated_value),
                "added_entries": len(mutated_value) - len(current_value),
                "pick_ratio": len(mutated_value) / len(current_value) if current_value else 0,
                "result_entries": len(mutated_value)
            }
            mutation_records.append(summary_row)

            individual = self._set_value_in_individual(individual, key, mutated_value)

            # Narrow valid_df progressively
            if key not in self.static_factors and mutated_value:
                column_name = key.split(".")[-1]
                if column_name in valid_df.columns:
                    valid_df = valid_df[valid_df[column_name].isin(mutated_value)]

        # Optionally attach mutation_records as self.last_mutation_summary:
        self.last_mutation_summary = pd.DataFrame(mutation_records)

        return (individual,), mutated





    def custom_crossover(self, ind1, ind2):
        # Semantic section crossover
        pass

    def transform_back(self, population, generation_no):
        # Return DataFrame
        pass

    def get_scores_and_attach(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Stub implementation to attach random fitness scores to a DataFrame.

        For now this assigns random values [0,1] to the score_acc, score_sample, 
        and score_entropy columns for every individual.

        Args:
            df (pd.DataFrame): The input DataFrame, must contain at least `payload`.

        Returns:
            pd.DataFrame: A copy of the input DataFrame with scores attached.
        """
        df = df.copy()
        df["score_acc"] = [random.uniform(0, 1) for _ in range(len(df))]
        df["score_sample"] = [random.uniform(0, 1) for _ in range(len(df))]
        df["score_entropy"] = [random.uniform(0, 1) for _ in range(len(df))]
        return df

    def evolve(self, old_gen_df, generation_no):
        pop = self.create_pop(old_gen_df)
        self.get_scores_and_attach(pop)
        offspring = self.toolbox.select(pop, len(pop))
        offspring = algorithms.varAnd(offspring, self.toolbox, cxpb=0.5, mutpb=0.2)
        return self.transform_back(offspring, generation_no)





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

    def _get_global_mutation_config(self):
        """
        Fetch the resolved DEAP config structure:
        - acc_type
        - control_branch_name
        - metadata (entire metadata dict)
        - mutation_config (nested inside metadata)
        """
        try:
            acc_type = self.branch_id
            control_branch_name = self.subbranch_id

            branch_config = self.gen_config.get(acc_type, {}).get(control_branch_name, {})

            metadata = branch_config.get("metadata", {})
            mutation_config = metadata.get("mutation_config", {})

            resolved = {
                "acc_type": acc_type,
                "control_branch_name": control_branch_name,
                "metadata": metadata,
                "mutation_config": mutation_config
            }

            logging.debug3(f"Resolved global mutation config: {resolved}")

            return resolved

        except Exception as e:
            logging.error(f"Error resolving mutation config for branch={self.branch_id} subbranch={self.subbranch_id}: {e}")
            return {
                "acc_type": self.branch_id,
                "control_branch_name": self.subbranch_id,
                "metadata": {},
                "mutation_config": {}
            }

    
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

    
    def _apply_must_not_contain(self, df, mutation_config):
        """
        Apply must_not_contain filters from mutation_config globally to wood_master_df.

        Args:
            df (pd.DataFrame): The dataframe to filter (typically wood_master_df).
            mutation_config (dict): The parsed mutation config dict.

        Returns:
            pd.DataFrame: Filtered dataframe with disallowed values removed.
        """
        filtered_df = df.copy()

        # Process static factors
        static_params = mutation_config.get("static_mutate_params", {})
        for key, params in static_params.items():
            must_not = params.get("must_not_contain", [])
            if must_not and key in filtered_df.columns:
                filtered_df = filtered_df[~filtered_df[key].isin(must_not)]

        # Process dependent factors
        dependent_params = mutation_config.get("dependent_mutate_params", {})
        for key, params in dependent_params.items():
            must_not = params.get("must_not_contain", [])
            if must_not and key in filtered_df.columns:
                filtered_df = filtered_df[~filtered_df[key].isin(must_not)]

        return filtered_df
    


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

        # Assume we work on first row for now — can be adjusted as needed:
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

        for section_name in ["static_mutate_params", "dependent_mutate_params"]:
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



