import logging


import yaml
import random
import hashlib
import time
import pandas as pd

from typing import List, Dict, Optional

from pathlib import Path

from app.tasks.TaskBase import TaskBase
from collections import defaultdict
from sqlalchemy import cast, Integer, select, update
from sqlalchemy.dialects.postgresql import JSONB

from copy import deepcopy

from app.utils.dataModels.Jobs.DoEJob import DoEJob, DOE_config, PrimaryData, SegmentationCfg, SecondaryData, PreProcessingCfg, ModelingCfg

from app.utils.SQL.models.production.api_ModellingResults import ModellingResults_Out
from app.utils.SQL.models.production.orm_ModellingResults import orm_ModellingResults

from app.utils.SQL.DBEngine import DBEngine

from app.utils.SQL.models.jobs.orm_DoEJobs import orm_DoEJobs
from app.utils.SQL.models.jobs.api_DoEJobs import DoEJobs_Out

from app.utils.SQL.models.production.api.api_WoodMasterPotential import WoodMasterPotential_Out



class TA28_0_DoECreatorOrchestrator(TaskBase):
    def setup(self):
        self.TARGET_BRANCHES = ["knn_acc", "rf_acc", "combo_01"]
        self.orchestration_dict = {}
        self.GENERATION_LIMIT = 50
        self.FINISHING_RATIO = 0.95
        self.GENERATION_SIZE = 30
        self.SLEEP_SECONDS = 10

        self.generation_progress_df = pd.DataFrame(columns=["branch", "subbranch", "gen", "count"])

        mock_data = [
            {"branch": "knn_acc", "subbranch": "branch_linear_01", "gen": 0, "count": 30},
            {"branch": "knn_acc", "subbranch": "branch_linear_01", "gen": 1, "count": 28},
            {"branch": "knn_acc", "subbranch": "branch_inverse_02", "gen": 0, "count": 30},
            {"branch": "knn_acc", "subbranch": "branch_inverse_02", "gen": 1, "count": 30},
            {"branch": "rf_acc", "subbranch": "branch_random_05", "gen": 0, "count": 29},
            {"branch": "rf_acc", "subbranch": "branch_random_05", "gen": 1, "count": 15},
            {"branch": "rf_acc", "subbranch": "branch_constant_01", "gen": 0, "count": 30},
            {"branch": "rf_acc", "subbranch": "branch_constant_01", "gen": 5, "count": 5},
            {"branch": "combo_01", "subbranch": "branch_inverse_03", "gen": 0, "count": 30},
            {"branch": "combo_01", "subbranch": "branch_inverse_03", "gen": 1, "count": 30},
            {"branch": "combo_01", "subbranch": "branch_random_03", "gen": 0, "count": 30},
            {"branch": "combo_01", "subbranch": "branch_random_03", "gen": 4, "count": 10},
        ]

        self.generation_progress_df = pd.DataFrame(mock_data)


    def run(self):
        try:
            logging.debug3("🚀 Step 1: Initialize orchestration dictionary")
            self.controller.update_message("Step 1: Initializing orchestration dictionary")
            self.woodMasterPotential_full = WoodMasterPotential_Out.fetch_all()
            
            self.define_DoE_Options()
            self.branch_instructions = self.load_branch_instructions()
            self.branch_config = self.create_branch_config()
        
            created_count = 0
            updated_count = 0

            while True:

                self.branch_config, summary_df = self.attach_current_gen()
                
                
                
                if self.controller.should_stop():
                    logging.debug3("🛑 Orchestration stopped by controller")
                    break
                    

                for branch_id in self.TARGET_BRANCHES:
                    for subbranch_id, config in self.branch_config[branch_id].items():

                        ratio = config.get("current_ratio", 0.0)
                        current_gen = config.get("current_gen", 0)

                        if current_gen >= self.GENERATION_LIMIT:
                            logging.debug2(f"🔚 Branch {branch_id}/{subbranch_id} reached generation limit ({current_gen}). Skipping.")
                            continue

                        if current_gen == 0:
                            logging.debug2(f"🌱 Branch {branch_id}/{subbranch_id} is starting at generation 0. Initializing initial population.")
                            jobs = self.create_initial_population(branch_id, subbranch_id, config)
                            
                            created_count_i, updated_count_i = store_or_update_jobs(jobs)
                            created_count += created_count_i
                            updated_count += updated_count_i

                            logging.debug2(f"✅ Branch {branch_id}/{subbranch_id} initialized with initial population")



                        if ratio >  self.FINISHING_RATIO:
                            pass


                        else:
                            logging.debug1(f"🔄 Branch {branch_id}/{subbranch_id} is waiting for completion")


                

            
            


        except Exception as e:
            logging.exception("❌ Task failed:")
            self.controller.finalize_failure(str(e))
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        logging.debug5("🧹 Running cleanup and profiling flush")
        self.flush_memory_logs()
        self.controller.archive_with_orm()

    def update_branch_generation_progress_df(self) -> pd.DataFrame:
        from app.utils.SQL.DBEngine import DBEngine
        from sqlalchemy import text
        import pandas as pd
        import logging

        all_results= pd.DataFrame(columns=["branch", "subbranch", "gen", "count"])

        for branch_id in self.TARGET_BRANCHES:
            session = DBEngine(db_key="jobs").get_session()

            try:
                sql = text("""
                    SELECT
                        :branch AS branch,
                        subbranch.key AS subbranch,
                        (branch.value->>'generation')::int AS gen,
                        COUNT(*) AS count
                    FROM "orm_DoEJobs",
                        jsonb_each("DEAP_metadata"->:branch) AS subbranch(key, value),
                        jsonb_each("DEAP_metadata"->:branch) AS branch(key, value)
                    WHERE (branch.value->>'generation') IS NOT NULL
                    GROUP BY subbranch.key, gen
                    ORDER BY subbranch.key, gen
                """)

                result = session.execute(sql, {"branch": branch_id}).fetchall()
                if result:
                    df = pd.DataFrame(result, columns=["branch", "subbranch", "gen", "count"])
                    all_results = pd.concat([all_results, df], ignore_index=True)
                else:
                    logging.debug2(f"🔍 No jobs found for branch {branch_id} in orchestration dict")

            except Exception as e:
                session.rollback()
                logging.error(f"❌ Failed to update generation progress for {branch_id}: {e}", exc_info=True)
                raise e
            finally:
                session.close()

        self.generation_progress_df = all_results

    def load_branch_instructions(self) -> dict:

        with open("app/config/DoE/DoE_Evolution.yaml", "r") as f:
            branch_instructions_raw = yaml.safe_load(f)
        



        return branch_instructions_raw
    

    def create_branch_config(self) -> dict:
        """
        Build full instruction tree:
        {
            parent_branch: {
                subbranch: {config...},
                ...
            },
            ...
        }
        """
        instruction_tree = defaultdict(dict)

        for parent_key in self.TARGET_BRANCHES:
            for group_key, group_cfg in self.branch_instructions.items():
                subbranches = group_cfg.get("control_branches", [])
                for sub in subbranches:
                    sub_cfg = {k: v for k, v in group_cfg.items() if k != "control_branches"}
                    instruction_tree[parent_key][sub] = sub_cfg

        return instruction_tree


    def attach_current_gen(self, threshold=0.95) -> dict:
        """
        Augments self.branch_config by adding current_gen, current_ratio, scheduled_gen to each subbranch.

        Returns:
            Same structure as self.branch_config:
            {
                branch: {
                    subbranch: {
                        ...existing_config,
                        current_gen: int,
                        current_ratio: float,
                        scheduled_gen: int
                    }
                }
            }
        """
        progress_df = self.generation_progress_df
        orchestration = deepcopy(self.branch_config)  # Keep original structure

        if progress_df.empty:
            logging.warning("⚠️ No jobs found. Assuming all branches start at gen=0.")
            for branch, sub_cfg in orchestration.items():
                for subbranch in sub_cfg:
                    orchestration[branch][subbranch].update({
                        "current_gen": 0,
                        "current_ratio": 0.0,
                        "scheduled_gen": 0
                    })
            return orchestration

        grouped = progress_df.groupby(["branch", "subbranch"])
        summary_rows = []

        for (branch, subbranch), group in grouped:
            max_gen = group["gen"].max()
            jobs_in_max_gen = group[group["gen"] == max_gen]["count"].sum()

            config = orchestration.get(branch, {}).get(subbranch, {})
            gen_size = config.get("generation_size", 10)

            ratio = jobs_in_max_gen / gen_size if gen_size else 0
            scheduled = max_gen + 1 if ratio >= threshold else max_gen

            config.update({
                "current_gen": int(max_gen),
                "current_ratio": float(round(ratio, 2)),
                "scheduled_gen": int(scheduled)
            })

            summary_rows.append({
                "gen": max_gen,
                "ratio": ratio
            })

        # Fill in 0s for missing entries
        for branch, sub_cfg in orchestration.items():
            for subbranch, cfg in sub_cfg.items():
                if "current_gen" not in cfg:
                    cfg.update({
                        "current_gen": 0,
                        "current_ratio": 0.0,
                        "scheduled_gen": 0
                    })

                    summary_rows.append({
                        "gen": 0,
                        "ratio": 0
                        })



        summary_df = pd.DataFrame()
        if summary_rows:
            df = pd.DataFrame(summary_rows)

            # Separate out gen 0
            gen0_df = df[df["gen"] == 0]
            df_nonzero = df[df["gen"] > 0]

            summary_data = []

            if not df_nonzero.empty:
                # Get unique non-zero generations
                available_gens = set(df_nonzero["gen"].unique())

                # Calculate basic stats (rounded to int)
                gen_stats = {
                    "min_gen": int(df_nonzero["gen"].min()),
                    "max_gen": int(df_nonzero["gen"].max()),
                    "avg_gen": int(round(df_nonzero["gen"].mean())),
                    "median_gen": int(df_nonzero["gen"].median())
                }

                for label, gen_val in gen_stats.items():
                    if gen_val in available_gens:
                        ratios = df_nonzero[df_nonzero["gen"] == gen_val]["ratio"]
                        summary_data.append({
                            "gen": gen_val,
                            "label": label,
                            "min_ratio": round(ratios.min(), 2),
                            "max_ratio": round(ratios.max(), 2),
                            "avg_ratio": round(ratios.mean(), 2),
                            "count": len(ratios)
                        })

            # Always include gen0 row
            if not gen0_df.empty:
                summary_data.append({
                    "gen": 0,
                    "label": "gen0",
                    "min_ratio": None,
                    "max_ratio": None,
                    "avg_ratio": None,
                    "count": len(gen0_df)
                })

            summary_df = pd.DataFrame(summary_data)
            logging.debug2(f"🧾 Orchestration Gen Summary:\n{summary_df.to_string(index=False)}")


        return orchestration, summary_df




    def initialize_orchestration_dict(self):
        """
        Initializes the orchestration dictionary with the latest known generation for each target branch and subbranch.

        For each `branch_id` in `self.TARGET_BRANCHES`, this function:
        - Queries the "orm_DoEJobs" table for the highest existing generation number for a fixed `subbranch_id`
        (currently hardcoded as "subbranch_01") using nested JSONB access on the "DEAP_metadata" column.
        - If no jobs are found for the given branch/subbranch combination, it initializes that path to generation 0.
        - If jobs are found, it sets the orchestration dict to the highest found generation number.
        - The result is stored in `self.orchestration_dict[branch_id][subbranch_id]`.

        This allows the orchestrator to resume or continue evolutionary search from the correct generation
        for each branch/subbranch path.

        Raises:
            Exception: Any database execution errors are caught and reraised after rolling back the session.
        """
        

        
        
        for branch_id in self.TARGET_BRANCHES:
            session = DBEngine(db_key="jobs").get_session()

            try:
                sql = text("""
                    SELECT
                        subbranch.key AS subbranch_id,
                        MAX((branch.value->>'generation')::int) AS max_generation
                    FROM "orm_DoEJobs",
                        jsonb_each("DEAP_metadata"->:branch) AS subbranch(key, value),
                        jsonb_each("DEAP_metadata"->:branch) AS branch(key, value)
                    WHERE (branch.value->>'generation') IS NOT NULL
                    GROUP BY subbranch.key

                """)



                result = session.execute(sql, {"branch": branch_id}).fetchall()

                # Store result as dict
                if result:
                    self.orchestration_dict[branch_id] = {
                        row["subbranch_id"]: row["max_generation"] for row in result
                    }
                    for sub_id, gen in self.orchestration_dict[branch_id].items():
                        logging.debug2(f"🔁 Found latest gen {gen} for branch {branch_id}/{sub_id}")
                else:
                    # No subbranches found → initialize to default
                    self.orchestration_dict[branch_id] = {"subbranch_01": 0}
                logging.debug2(f"🌱 Seeded gen 0 for branch {branch_id}/subbranch_01")

            except Exception as e:
                session.rollback()
                logging.error(f"❌ Failed to initialize orchestration dict for {branch_id}: {e}", exc_info=True)
                raise e
            finally:
                session.close() 


    def define_DoE_Options(self):
        def _load_yaml_keys(yaml_path: str) -> list[str]:
            with open(Path(yaml_path), "r") as f:
                data = yaml.safe_load(f)
            return list(data.keys())


        self.STATIC_FACTORS = {
            "primary_data.sourceNo": {
                "values": self.woodMasterPotential_full["sourceNo"].unique().tolist(),
                "multi": True
            },
            #"primary_data.maxShots": {
            #    "values": [500],  # Or pull from YAML if it's dynamic
            #    "multi": False
            #},

            "primary_data.noShotsRange": {
                "values": [[None], [[2, 15]], [[2, 100]], [[1, 5]], [[10,25]]],
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

        self.DEPENDENT_FACTORS = {
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


    def generate_random_doe_config(self, constraints: dict = None) -> dict:
        config = {
            "primary_data": {},
            "secondary_data": {},
            "preprocessing": {},
            "modeling": {},
            "segmentation": {}
        }

        df = self.woodMasterPotential_full.copy()

        # Apply constraints if provided
        if constraints:
            include = constraints.get("include", {})
            exclude = constraints.get("exclude", {})

            for field, allowed_values in include.items():
                section, col = field.split(".")
                df = df[df[col].isin(allowed_values)]

            for field, blocked_values in exclude.items():
                section, col = field.split(".")
                df = df[~df[col].isin(blocked_values)]

        # 1. Handle static factors
        for key, cfg in self.STATIC_FACTORS.items():
            section, field = key.split(".")
            choices = cfg["values"]

            # Apply filtered candidates if they exist in constraints
            if field in df.columns and not df.empty:
                candidates = df[field].dropna().unique().tolist()
                if candidates:
                    choices = [val for val in choices if val in candidates]

            if cfg["multi"]:
                value = random.sample(choices, random.randint(1, len(choices))) if choices else []
            else:
                value = random.choice(choices) if choices else None

            config[section][field] = value

        # 2. Handle dependent factors
        filled_keys = set(self.STATIC_FACTORS.keys())

        def resolve_dependencies(target_key):
            if target_key in filled_keys:
                return

            depends = self.DEPENDENT_FACTORS[target_key]["depends_on"]
            for dep_key in depends:
                resolve_dependencies(dep_key)

            section, field = target_key.split(".")
            df_filtered = df.copy()

            for dep in depends:
                dep_section, dep_field = dep.split(".")
                dep_val = config[dep_section].get(dep_field)
                if dep_val is not None:
                    if isinstance(dep_val, list):
                        df_filtered = df_filtered[df_filtered[dep_field].isin(dep_val)]
                    else:
                        df_filtered = df_filtered[df_filtered[dep_field] == dep_val]

            candidates = df_filtered[field].dropna().unique().tolist() if not df_filtered.empty else []

            multi = self.DEPENDENT_FACTORS[target_key]["multi"]
            if not candidates:
                config[section][field] = []
            elif multi:
                config[section][field] = random.sample(candidates, random.randint(1, len(candidates)))
            else:
                config[section][field] = random.choice(candidates)

            filled_keys.add(target_key)

        for key in self.DEPENDENT_FACTORS:
            resolve_dependencies(key)

        return config



    def create_initial_population(self, branch_id: str, subbranch_id: str, config: dict) -> List[DoEJob]:
        from app.tasks.TA27_DesignOfExperiments.TA27_A_DoEJobGenerator import TA27_A_DoEJobGenerator

        def _flatten_doe_row(new_row):
            """
            Converts a nested DoE row (new format) to flat legacy-style format.
            Returns a dictionary with all expected fields.
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
                "noShotsRange": new_row["primary_data"].get("noShotsRange", [30]),
            }


        generation_size = config.get("generation_size", 30)
        configs = []
        flat_rows = []

        for _ in range(generation_size):
            base_cfg = self.generate_random_doe_config()

            # Inject evolution config into modeling
            base_cfg["modeling"].update({
                k: v for k, v in config.items() if k in [
                    "mutation_strategy", "mutation_func", "allow_multi_mutation",
                    "fallback_on_invalid", "corridor", "max_generations",
                    "generation_size", "elite_size", "constraints"
                ]
            })


            flat_cfg = _flatten_doe_row(base_cfg)

            flat_cfg["branch_id"] = branch_id
            flat_cfg["subbranch_id"] = subbranch_id
            flat_cfg["generation"] = 0
            flat_cfg["origin"] = "init"
            flat_cfg["created_by"] = "TA28_DoECreator"

            flat_rows.append(flat_cfg)


        # Add DoE_UUID to each row
        for row in flat_rows:
            row["DoE_UUID"] = generate_doe_uuid(row)


        df = pd.DataFrame(flat_rows)

        # Generate DoEJob objects
        jobs = TA27_A_DoEJobGenerator.generate(df=df, template_path="app/config/templates/DoE_job_template.yaml")
        job_df = pd.DataFrame([job.to_sql_row() for job in jobs])
        
        
        return job_df


    def mutate_generation(self, branch_id: str, subbranch_id: str,) -> pd.DataFrame:
        

        gen = self.branch_config[branch_id][subbranch_id].get("current_gen")
        
        
        old_DoEs = self.get_DoEs_of_last_gen(branch_id=branch_id, subbranch_id=subbranch_id, gen = gen)
        
        def attach_weighted_scores(doe_df: pd.DataFrame, fraq_weights: dict, resolution_weights: dict) -> pd.DataFrame:
            """
            Attach weighted scores to each DoE config row.

            Args:
                doe_df (pd.DataFrame): DataFrame of DoEs for the current generation. Must contain 'DoE_UUID'.
                fraq_weights (dict): Mapping of FRAQ values to their weights (from YAML).
                resolution_weights (dict): Mapping of scope/label resolution levels to weights.

            Returns:
                pd.DataFrame: Input DataFrame with a new 'score' column.
            """
            # TODO: Query orm_ModellingResults based on DoE_UUID
            # TODO: Filter relevant FRAQs, scopes
            # TODO: Apply fraq_weights * resolution_weights
            # TODO: Aggregate (e.g., weighted mean or quartile) per DoE_UUID
            pass


        


    def get_DoEs_of_last_gen(self, branch_id: str, subbranch_id: str, gen: int) -> pd.DataFrame:
        """
        Fetch all DoE job rows from a specific generation for a given (branch_id, subbranch_id).

        Args:
            branch_id (str): The branch name used in the DEAP_metadata JSONB.
            subbranch_id (str): The subbranch name under the branch.
            gen (int): The generation number to filter for.

        Returns:
            pd.DataFrame: Matching DoEJobs as flat rows.
        """
        session = DBEngine(db_key="jobs").get_session()

        try:
            stmt = select(orm_DoEJobs).where(
                cast(
                    orm_DoEJobs.DEAP_metadata[branch_id][subbranch_id]["generation"].astext,
                    Integer
                ) == gen
            )

            results = session.execute(stmt).scalars().all()

            if not results:
                logging.info(f"ℹ️ No jobs found for ({branch_id}, {subbranch_id}) at gen={gen}")
                return pd.DataFrame()

            df = pd.DataFrame([job.to_sql_row() for job in results])
            return df

        except Exception as e:
            logging.error(f"❌ Failed to fetch jobs for ({branch_id}, {subbranch_id}) @ gen {gen}: {e}", exc_info=True)
            return pd.DataFrame()

        finally:
            session.close()

    def select_elite_and_tournament(doe_df: pd.DataFrame, n_elite: int, m_tournament: int, tournament_k: int) -> List[dict]:
        """
        Select top-n elite and m additional candidates via tournament selection.

        Args:
            doe_df (pd.DataFrame): Scored DoE DataFrame, must contain 'score' column.
            n_elite (int): Number of elite individuals to keep unchanged.
            m_tournament (int): Number of candidates to select via tournament.
            tournament_k (int): Number of contenders per tournament.

        Returns:
            List[dict]: Selected DoE configs to serve as parents for mutation.
        """
        # TODO: Sort by score and pick top-n
        # TODO: Run m tournaments of size k, pick best from each
        pass



    def mutate_configs(parents: List[dict], mutation_func: Callable, allow_multi: bool, constraints: dict) -> List[dict]:
        """
        Generate new DoE configs by mutating the parent configs.

        Args:
            parents (List[dict]): List of parent configs selected from previous generation.
            mutation_func (Callable): Mutation strength function (based on generation).
            allow_multi (bool): Whether multiple mutations per config are allowed.
            constraints (dict): Dict of inclusion/exclusion rules to enforce.

        Returns:
            List[dict]: List of new mutated DoE config dicts.
        """
        # TODO: For each parent, generate new config by mutating random field(s)
        # TODO: Respect constraints (e.g., forced inclusion/exclusion)
        # TODO: Optionally apply multiple mutations if allowed
        pass

    def create_next_generation_jobs(mutated_configs: List[dict], branch_id: str, subbranch_id: str, new_gen: int) -> pd.DataFrame:
        """
        Flatten, UUID-hash, and prepare new DoE jobs for insertion.

        Args:
            mutated_configs (List[dict]): List of mutated config dicts.
            branch_id (str): Parent branch name.
            subbranch_id (str): Subbranch under which these configs were evolved.
            new_gen (int): Generation number for the new configs.

        Returns:
            pd.DataFrame: Job rows ready to be stored in the DB (including DoE_UUID, DEAP_metadata, etc.)
        """
        # TODO: Flatten nested DoE dicts
        # TODO: Sort lists and hash to get DoE_UUID
        # TODO: Build DEAP_metadata with generation info
        pass

    def store_or_update_jobs(job_df: pd.DataFrame) -> Tuple[int, int]:
        """
        Store new jobs or update DEAP metadata for existing ones.

        Args:
            job_df (pd.DataFrame): DataFrame of jobs including 'DoE_UUID' and 'DEAP_metadata'.

        Returns:
            Tuple[int, int]: Number of jobs inserted, number of jobs updated.
        """
        # TODO: Check existence by UUID
        # TODO: Insert or merge DEAP_metadata accordingly
        # TODO: Commit to DB and return counts
        pass



def generate_doe_uuid(row: dict) -> str:
    """Generates a deterministic DoE_UUID from a config row.
    Ensures all values are lists and sorts both keys and their contents.
    """
    normalized_row = {}

    for k, v in row.items():
        if isinstance(v, list):
            normalized_row[k] = sorted(v)  # Sort the list itself
        else:
            normalized_row[k] = [v]  # Wrap scalar in list

    base_str = str(sorted(normalized_row.items()))  # Sort key-value pairs
    return "DoE_" + hashlib.sha1(base_str.encode()).hexdigest()[:10]



def store_or_update_jobs(job_df):
    session = DBEngine(db_key="jobs").get_session()

    created_count = 0
    updated_count = 0

    try:
        for job in job_df.to_dict(orient="records"):
            job_uuid = job["job_uuid"]

            # Check if job already exists
            existing = session.execute(
                select(orm_DoEJobs).where(orm_DoEJobs.job_uuid == job_uuid)
            ).scalar_one_or_none()

            if not existing:
                # Insert as new row
                session.add(orm_DoEJobs(**job))
                created_count += 1
            else:
                # Merge DEAP_metadata
                existing_metadata = existing.DEAP_metadata or {}
                new_metadata = job.get("DEAP_metadata", {})

                for branch, subbranches in new_metadata.items():
                    if branch not in existing_metadata:
                        existing_metadata[branch] = subbranches
                    else:
                        existing_metadata[branch].update(subbranches)

                # Update the row in DB
                session.execute(
                    update(orm_DoEJobs)
                    .where(orm_DoEJobs.job_uuid == job_uuid)
                    .values(DEAP_metadata=existing_metadata)
                )
                updated_count += 1

        session.commit()
        

    except Exception as e:
        session.rollback()
        logging.error(f"❌ Failed to store/update DoE_Jobs: {e}", exc_info=True)
        raise e
    finally:
        session.close()
        return created_count, updated_count
