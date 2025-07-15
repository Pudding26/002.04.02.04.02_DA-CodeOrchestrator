import logging


import yaml
import random
import hashlib
import time
import pandas as pd

from typing import List, Dict, Optional, Callable, Union, Tuple
from queue import Empty

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



from app.tasks.TA28_DoECreator.TA28_A_Initiater import TA28_A_Initiater
from app.tasks.TA28_DoECreator.TA28_B_Mutator import TA28_B_Mutator

from app.tasks.TA28_DoECreator.utils.doe_factor_config import define_DoE_Options
from app.tasks.TA28_DoECreator.utils.store_or_update_jobs import store_or_update_jobs


FINISHING_RATIO = 0.95  # Ratio threshold to consider a generation finished



class TA28_0_DoECreatorOrchestrator(TaskBase):
    def setup(self):
        logging.info("🧵 Setting up DoE Creator Orchestrator")
        self.TARGET_BRANCHES = ["knn_acc", "rf_acc", "combo_01"]
        self.TARGET_BRANCHES = self.TARGET_BRANCHES[0:1]
        self.orchestration_dict = {}
        self.SLEEP_SECONDS = 10


        self.generation_progress_df = pd.DataFrame(columns=["branch", "subbranch", "gen", "count"])

    def run(self):
        """
        Main orchestration loop for DoE evolutionary optimization.

        Responsibilities:
        - Monitor generation progress across all branches/subbranches.
        - Initialize populations if they do not yet exist (gen == -1).
        - Trigger mutation workflow for branches ready to advance.
        - Handle backoff, retries, and graceful shutdown.

        This orchestrator itself does NOT contain business logic for population creation or mutation.
        Those responsibilities are delegated to TA28_A_Initiater and TA28_B_Mutator.
        """
        try:
            logging.debug3("🚀 Step 1: Initialize orchestration dictionary and state")
            self.controller.update_message("Step 1: Initializing orchestration dictionary")
            self.woodMasterPotential_full = WoodMasterPotential_Out.fetch_all()
            
            self.STATIC_FACTORS, self.DEPENDENT_FACTORS = define_DoE_Options(self.woodMasterPotential_full)  # Derive vocabularies for DoE factors
            self.branch_instructions = self.load_branch_instructions()  # Load canonical YAML template
            self.branch_config = self.create_branch_config()  # Setup orchestration state

            # Constants for retry/backoff loop
            self.MAX_SLEEP_SECONDS = 600
            self.BASE_SLEEP_SECONDS = 10

            created_count_total = 0
            updated_count_total = 0
            backoff = 0
            loop_no = 0

            # Main orchestration loop
            while True:
                self.generation_progress_df = self.update_branch_generation_progress_df()
                self.branch_config, summary_df = self.attach_current_gen()

                created_count = 0
                updated_count = 0

                if self.controller.should_stop():
                    logging.debug3("🛑 Orchestration stopped by controller")
                    break

                for branch_id in self.TARGET_BRANCHES:
                    for subbranch_id, config in self.branch_config[branch_id].items():
                        ratio = config.get("current_ratio")
                        current_gen = config.get("current_gen")

                        # Check generation limit
                        if current_gen >= config.get("max_generations"):
                            logging.debug2(f"🔚 Branch {branch_id}/{subbranch_id} reached generation limit ({current_gen}). Skipping.")
                            continue

                        # Initialize population if needed
                        if current_gen == -1:
                            logging.debug2(f"🌱 Branch {branch_id}/{subbranch_id} is starting at generation 0. Initializing initial population.")
                            config["current_gen"] = 0

                            initiater = TA28_A_Initiater(
                                wood_master_df=self.woodMasterPotential_full,
                                static_factors=self.STATIC_FACTORS,
                                dependent_factors=self.DEPENDENT_FACTORS
                            )
                            initial_jobs_df = initiater.create_initial_population(
                                branch_id=branch_id,
                                subbranch_id=subbranch_id,
                                branch_config=config
                            )
                            

                            created_count_i, updated_count_i = store_or_update_jobs(initial_jobs_df)
                            created_count += created_count_i
                            updated_count += updated_count_i

                            config["gen0_created"] = True
                            logging.debug2(f"✅ Branch {branch_id}/{subbranch_id} initialized with initial population")

                        # Trigger mutation for ready branches
                        #if config.get("scheduled_gen") > config.get("current_gen"):
                        if 1 == 1:
                            last_gen_doe_df = self.get_DoEs_of_last_gen(
                                branch_id=branch_id,
                                subbranch_id=subbranch_id,
                                gen=current_gen
                            )
                            #last_gen_doe_df = last_gen_doe_df[last_gen_doe_df["modeler_status"] == "DONE"]
                            TA28_B_Mutator(
                                last_gen_doe_df=last_gen_doe_df,
                                wood_master_df=self.woodMasterPotential_full,
                                static_factors=self.STATIC_FACTORS,
                                dependent_factors=self.DEPENDENT_FACTORS,
                                mutation_config_raw=self.branch_config,
                                branch_id=branch_id,
                                subbranch_id=subbranch_id,
                            ).mutate_generation()

                            # Call to TA28_B_Mutator will go here:
                            pass

                        else:
                            logging.debug1(f"🔄 Branch {branch_id}/{subbranch_id} is waiting for completion")

                created_count_total += created_count
                updated_count_total += updated_count

                logging.debug5(f"📝 Created {created_count} jobs, updated {updated_count} jobs in this run")
                if created_count == 0 and updated_count == 0:
                    sleep_time = min(self.BASE_SLEEP_SECONDS * (2 ** backoff), self.MAX_SLEEP_SECONDS)
                    logging.info(f"🕒 No jobs — sleeping {sleep_time}s (backoff={backoff})")
                    time.sleep(sleep_time)
                    backoff += 1
                    loop_no += 1
                    continue

                backoff = 0  # reset backoff if work was done





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



### Utils

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


    def update_branch_generation_progress_df(self) -> pd.DataFrame:
        """
        Query the DoEJobs database and construct a progress DataFrame showing
        the count of completed (DONE) and failed (FAILED) jobs for each
        branch, subbranch, and generation.

        Returns:
            pd.DataFrame: A DataFrame with columns:
                - branch: Branch identifier (str)
                - subbranch: Subbranch identifier (str)
                - gen: Generation number (int)
                - count: Total finished jobs (DONE + FAILED)
                - count_done: Jobs with status DONE
                - count_failed: Jobs with status FAILED
        """
        from app.utils.SQL.DBEngine import DBEngine
        from sqlalchemy import text
        import pandas as pd
        import logging

        all_results = pd.DataFrame(columns=["branch", "subbranch", "gen", "count", "count_done", "count_failed"])

        for branch_id in self.TARGET_BRANCHES:
            session = DBEngine(db_key="jobs").get_session()

            try:
                sql = text("""
                    SELECT
                        :branch AS branch,
                        subbranch.key AS subbranch,
                        (branch.value->>'generation')::int AS gen,
                        COUNT(*) FILTER (WHERE "modeler_status" IN ('DONE', 'FAILED')) AS count,
                        COUNT(*) FILTER (WHERE "modeler_status" = 'DONE') AS count_done,
                        COUNT(*) FILTER (WHERE "modeler_status" = 'FAILED') AS count_failed
                    FROM "DoEJobs",
                        jsonb_each("DEAP_metadata"->:branch) AS subbranch(key, value),
                        jsonb_each("DEAP_metadata"->:branch) AS branch(key, value)
                    WHERE (branch.value->>'generation') IS NOT NULL
                    GROUP BY subbranch.key, gen
                    ORDER BY subbranch.key, gen
                """)

                result = session.execute(sql, {"branch": branch_id}).fetchall()
                if result:
                    df = pd.DataFrame(result, columns=["branch", "subbranch", "gen", "count", "count_done", "count_failed"])
                    all_results = pd.concat([all_results, df], ignore_index=True)
                else:
                    logging.debug2(f"🔍 No jobs found for branch {branch_id} in orchestration dict")

            except Exception as e:
                session.rollback()
                logging.error(f"❌ Failed to update generation progress for {branch_id}: {e}", exc_info=True)
                raise e

            finally:
                session.close()

        return all_results


    def attach_current_gen(self, threshold=FINISHING_RATIO) -> dict:
        """
        Attach current generation progress metadata to `self.branch_config`.

        For each branch/subbranch, it computes:
        - `current_gen`: max generation number with any jobs present
        - `current_ratio`: ratio of jobs DONE/FAILED to generation size
        - `scheduled_gen`: next generation number if threshold met
        - `done_count`: DONE jobs in current generation
        - `failed_count`: FAILED jobs in current generation

        Also returns a `summary_df` capturing key stats across branches:
        - min, max, avg, median generation
        - corresponding ratios

        Returns:
            tuple:
                - Updated `orchestration` dict with branch/subbranch annotations
                - Summary DataFrame with overall generation progress summary
        """
        progress_df = self.generation_progress_df
        orchestration = deepcopy(self.branch_config)
        count = 0

        if progress_df.empty:
            logging.warning("⚠️ No jobs found. Assuming all branches start at gen=0.")
            for branch, sub_cfg in orchestration.items():
                for subbranch in sub_cfg:
                    orchestration[branch][subbranch].update({
                        "current_gen": -1,
                        "current_ratio": 0.0,
                        "scheduled_gen": 0,
                        "done_count": 0,
                        "failed_count": 0
                    })
                    count += 1

            summary_df = pd.DataFrame([{
                "gen": 0,
                "label": "gen0",
                "min_ratio": None,
                "max_ratio": None,
                "avg_ratio": None,
                "count": count
            }])

            return orchestration, summary_df

        grouped = progress_df.groupby(["branch", "subbranch"])
        summary_rows = []

        for (branch, subbranch), group in grouped:
            max_gen = group["gen"].max()
            gen_group = group[group["gen"] == max_gen]

            jobs_in_max_gen = gen_group["count"].sum()
            done_count = gen_group["count_done"].sum() if "count_done" in gen_group else 0
            failed_count = gen_group["count_failed"].sum() if "count_failed" in gen_group else 0

            config = orchestration.get(branch, {}).get(subbranch, {})
            gen_size = config.get("generation_size", 10)

            ratio = jobs_in_max_gen / gen_size if gen_size else 0
            scheduled = max_gen + 1 if ratio >= threshold else max_gen

            config.update({
                "current_gen": int(max_gen),
                "current_ratio": round(ratio, 2),
                "scheduled_gen": int(scheduled),
                "done_count": int(done_count),
                "failed_count": int(failed_count)
            })

            summary_rows.append({"gen": max_gen, "ratio": ratio})

        # Fill defaults for missing entries
        for branch, sub_cfg in orchestration.items():
            for subbranch, cfg in sub_cfg.items():
                if "current_gen" not in cfg:
                    cfg.update({
                        "current_gen": -1,  # Force initialization for missing entries
                        "current_ratio": 0.0,
                        "scheduled_gen": 0,
                        "done_count": 0,
                        "failed_count": 0
                    })
                    summary_rows.append({"gen": 0, "ratio": 0})

        # Build summary DataFrame
        summary_df = pd.DataFrame()
        if summary_rows:
            df = pd.DataFrame(summary_rows)
            gen0_df = df[df["gen"] == 0]
            df_nonzero = df[df["gen"] > 0]

            summary_data = []

            if not df_nonzero.empty:
                available_gens = set(df_nonzero["gen"].unique())
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
        Fetch all DoE job rows from a specific generation for a given (branch_id, subbranch_id),
        where modeler_status = 'DONE', and return as a complete DataFrame without using to_sql_row.
        """
        session = DBEngine(db_key="jobs").get_session()

        try:
            stmt = select(orm_DoEJobs).where(
                cast(
                    orm_DoEJobs.DEAP_metadata[branch_id][subbranch_id]["generation"].astext,
                    Integer
                ) == gen,
                #orm_DoEJobs.modeler_status.in_(['DONE', 'FAILED'])

            )

            results = session.execute(stmt).scalars().all()

            if not results:
                logging.info(f"ℹ️ No DONE jobs found for ({branch_id}, {subbranch_id}) at gen={gen}")
                return pd.DataFrame()

            rows = []
            for job in results:
                row = job.__dict__.copy()
                row.pop('_sa_instance_state', None)  # Remove SQLAlchemy's internal metadata
                rows.append(row)

            df = pd.DataFrame(rows)
            logging.debug2(f"✅ Fetched {len(df)} DONE jobs for ({branch_id}, {subbranch_id}) gen={gen}")
            return df

        except Exception as e:
            logging.error(f"❌ Failed to fetch DONE jobs for ({branch_id}, {subbranch_id}) @ gen {gen}: {e}", exc_info=True)
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


