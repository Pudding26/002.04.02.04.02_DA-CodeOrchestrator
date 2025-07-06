import logging
import yaml
from pathlib import Path



from app.tasks.TaskBase import TaskBase
from collections import defaultdict
from sqlalchemy import cast, Integer
from sqlalchemy.dialects.postgresql import JSONB



from app.utils.SQL.models.production.api_ModellingResults import ModellingResults_Out
from app.utils.SQL.models.production.orm_ModellingResults import orm_ModellingResults
from app.utils.SQL.models.jobs.orm_DoEJobs import orm_DoEJobs
from app.utils.SQL.models.jobs.api_DoEJobs import DoEJobs_Out




class TA28_0_DoECreatorOrchestrator(TaskBase):
    def setup(self):
        self.TARGET_BRANCHES = ["knn_acc_01", "rf_acc_01", "combo_01"]
        self.orchestration_dict = {}
        self.GENERATION_LIMIT = 50
        self.GENERATION_SIZE = 30
        self.SLEEP_SECONDS = 10

    def run(self):
        try:
            logging.debug3("🚀 Step 1: Initialize orchestration dictionary")
            self.controller.update_message("Step 1: Initializing orchestration dictionary")
            self.initalize_orchestration_dict()



            while self.orchestration_dict:
                    for branch_id, current_gen in list(self.orchestration_dict.items()):
                        if current_gen >= self.GENERATION_LIMIT:
                            print(f"🏁 Finished {branch_id} (gen {current_gen})")
                            del self.orchestration_dict[branch_id]
                            continue


                        


                        df = pd.read_sql_query(
                            f"""
                            SELECT * FROM "DoEJobs"
                            WHERE metadata ->> 'branch_id' = %s
                            AND (metadata ->> 'generation')::int = %s
                            """,
                            con=engine,
                            params=[branch_id, str(current_gen)]
                        )

                        if df[df["status"] == "done"].shape[0] < self.GENERATION_SIZE:
                            print(f"⏳ Waiting: {branch_id} gen {current_gen}")
                            continue

                        print(f"✅ Gen {current_gen} done for {branch_id} → evolving...")

                        parents_df = select_elites_and_last_gen(df, top_k=15)

                        encoder = DoEEncoder()
                        encoder.fit(parents_df, list_fields=["genus", "species", "family", "sourceNo", "woodType"])
                        X = encoder.transform(parents_df)
                        y = parents_df["combined_score"]

                        model = SurrogateModel()
                        model.fit(X, y)

                        evolver = EvolutionSearch(model=model, encoder=encoder)
                        new_X = evolver.generate(n=self.GENERATION_SIZE)
                        new_payloads = encoder.inverse_transform(new_X)

                        for payload in new_payloads:
                            create_doe_job(
                                payload=payload,
                                metadata={
                                    "branch_id": branch_id,
                                    "generation": current_gen + 1,
                                    "origin": f"{branch_id}_gen{current_gen}"
                                }
                            )

                        self.orchestration_dict[branch_id] += 1

                    print(f"⏲ Sleeping {self.SLEEP_SECONDS}s before next check...")
                    time.sleep(self.SLEEP_SECONDS)


   

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



def initialize_orchestration_dict(self):
    """
    Initializes the orchestration dictionary with the latest known generation for each target branch and subbranch.

    For each `branch_id` in `self.TARGET_BRANCHES`, this function:
    - Queries the "DoEJobs" table for the highest existing generation number for a fixed `subbranch_id`
      (currently hardcoded as "subbranch_01") using nested JSONB access on the "DEAP_metadata" column.
    - If no jobs are found for the given branch/subbranch combination, it initializes that path to generation 0.
    - If jobs are found, it sets the orchestration dict to the highest found generation number.
    - The result is stored in `self.orchestration_dict[branch_id][subbranch_id]`.

    This allows the orchestrator to resume or continue evolutionary search from the correct generation
    for each branch/subbranch path.

    Raises:
        Exception: Any database execution errors are caught and reraised after rolling back the session.
    """
        
        from app.utils.SQL.DBEngine import DBEngine
        from sqlalchemy import text
        
        for branch_id in self.TARGET_BRANCHES:
            session = DBEngine(db_key="jobs").get_session()

            try:
                sql = text("""
                    SELECT
                        subbranch.key AS subbranch_id,
                        MAX((branch.value->>'generation')::int) AS max_generation
                    FROM "DoEJobs",
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
                    self.orchestration_dict[branch_id] = {"root_branch": 0}
                logging.debug2(f"🌱 Seeded gen 0 for branch {branch_id}/root_branch")

            except Exception as e:
                session.rollback()
                logging.error(f"❌ Failed to initialize orchestration dict for {branch_id}: {e}", exc_info=True)
                raise e
            finally:
                session.close() 




    def perpare_DoEJobs(self):
        """
        Prepare DoEJobs for further processing.
        """
        # Convert raw DoEJobs to DataFrame
        df = pd.DataFrame([job.to_dict() for job in self.DoEJobs_raw])
        
        # Perform any necessary preprocessing here
        # For example, filter, rename columns, etc.
        
        return df
    
    def prepare_modellingResults(self):
        """
        Prepare ModellingResults for further processing.
        """
        # Convert raw ModellingResults to DataFrame
        df = pd.DataFrame([result.to_dict() for result in self.modellingResults_raw])
        
        # Perform any necessary preprocessing here
        # For example, filter, rename columns, etc.
        
        return df
    


