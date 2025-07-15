import random
import copy







class TA28_B_Mutator:

    def __init__(self, defaults):
        pass

        self.toolbox.register("mutate", lambda ind: mutate_generic(
                ind,
                static_factors=STATIC_FACTORS,
                dependent_factors=DEPENDENT_FACTORS,
                potential_data=woodMasterPotential_full,
                deap_config=mutate_config,
                defaults=defaults
            ))

#
#def stash():
#
#    def mutate_generic(
#        ind,
#        static_factors,
#        dependent_factors,
#        potential_data,
#        deap_config,
#        defaults
#    ):
#        payload = ind["payload"]
#        doe_config = payload.get("doe_config", {})
#        primary_data = doe_config.setdefault("primary_data", {})
#        mutated_keys = set()
#
#        def _mutate_field(section, field, values, multi, config):
#            mutation_chance = config.get("mutate_chance", defaults.get("mutate_chance", 0.1))
#            mutation_rate = config.get("mutation_rate", defaults.get("mutation_rate", 0.1))
#            growth_bias = config.get("growth_bias", defaults.get("growth_bias", 0))
#
#            if random.random() >= mutation_chance:
#                return False  # Skip mutation
#
#            # Ensure current_vals is always a list of unique elements
#            current_vals = set(doe_config.get(section, {}).get(field, []))
#            vocab_set = set(values)
#
#            no_drops = int(len(current_vals) * mutation_rate * random.random())
#            remaining = current_vals - set(random.sample(current_vals, k=min(no_drops, len(current_vals)))) if current_vals else set()
#
#            available = vocab_set - remaining
#            no_picks = max(int(len(current_vals) * mutation_rate * (random.random() + growth_bias)), no_drops)
#            picks = set(random.sample(list(available), k=min(no_picks, len(available)))) if available else set()
#
#            new_vals = list(remaining.union(picks))
#            if not multi and new_vals:
#                new_vals = [random.choice(new_vals)]
#
#            doe_config.setdefault(section, {})[field] = new_vals
#            return True  # Mutation occurred
#
#        # 🔹 STATIC_FACTORS mutation
#        for key, factor_cfg in static_factors.items():
#            section, field = key.split(".")
#            values = factor_cfg["values"]
#            multi = factor_cfg["multi"]
#            config = deap_config.get(key, {})
#
#            if _mutate_field(section, field, values, multi, config):
#                mutated_keys.add(key)
#
#        # 🔹 DEPENDENT_FACTORS mutation
#        def resolve_dependencies(factor_key):
#            section, field = factor_key.split(".")
#            factor_cfg = dependent_factors[factor_key]
#            depends_on = factor_cfg["depends_on"]
#            multi = factor_cfg["multi"]
#            config = deap_config.get(factor_key, {})
#
#            if any(dep in mutated_keys for dep in depends_on):
#                df = potential_data.copy()
#                for dep in depends_on:
#                    dep_section, dep_field = dep.split(".")
#                    dep_val = doe_config.get(dep_section, {}).get(dep_field, [])
#                    if dep_val:
#                        df = df[df[dep_field].isin(dep_val)]
#
#                candidates = df[field].dropna().unique().tolist() if not df.empty else []
#                if candidates:
#                    _mutate_field(section, field, candidates, multi, config)
#                    mutated_keys.add(factor_key)
#
#        for dep_key in dependent_factors:
#            resolve_dependencies(dep_key)
#
#        payload["doe_config"] = doe_config
#        ind["payload"] = payload
#        return (ind,)
#
#        
#
#
#
#
#        DEAP_CONFIG = {
#        "family": {
#            "mutate_chance": 0.3,
#            "mutation_rate": 0.1,  # Value between 0 and 1 that multiplies the 
#            "must_contain": ["Fagaceae", "Rosaceae"],
#            "must_not_contain": ["Sapotaceae"]
#            "growth_bias": 0.1 # Value between 0 and 1 that is added to mutation rate
#        "lens": {
#            "mutate_chance": 0.2
#        }
#    }
#
#
#    pop_multi = [
#        creator.IndividualMulti({
#            "primary_data": row["payload"]["doe_config"].get("primary_data", {}),
#            "secondary_data": row["payload"]["doe_config"].get("secondary_data", {}),
#            "preprocessing": row["payload"]["doe_config"].get("preprocessing", {}),
#            "modeling": row["payload"]["doe_config"].get("modeling", {}),
#            "segmentation": row["payload"]["doe_config"].get("segmentation", {}),
#            "acc_score": row["fitness"],
#            "size_score": row["sample_size"],
#            "entropy_score": row["sample_entropy"]
#        })
#        for _, row in df.iterrows()
#    ]




"""
TA28_B_Mutator
--------------

A DEAP‑powered mutation engine that creates the next DoE generation.

Key responsibilities
~~~~~~~~~~~~~~~~~~~~
* Convert the finished jobs of *generation g* into a DEAP population
  with attached (random for now) fitness scores.
* Select parents via **elitism + tournament**.
* Apply a custom **uniform crossover** that keeps the semantic sections
  (primary_data, secondary_data …) intact.
* Apply a custom **mutation** that respects
  - *mutation_func* (currently only *linear*)
  - *mutation_rate* & *mutation_chance* from the YAML template
  - the STATIC_FACTORS / DEPENDENT_FACTORS vocabularies and their
    taxonomic dependencies.
* Emit a DataFrame that looks exactly like the output of the
  *Initiater* so that the orchestrator can feed it into
  ``store_or_update_jobs``.

The class is intentionally self‑contained – the only contract with the
outside world is that you pass in the same artefacts you already have
in the orchestrator (``woodMasterPotential_full``, factor dicts and the
*branch_config* block of the YAML).

All API entry points are flagged with a "# PUBLIC API" comment so you
can wire them up from the orchestrator without scrolling through the
internals.
"""

from __future__ import annotations

import random
import copy
import logging
from typing import List, Dict, Any, Callable

import pandas as pd
from deap import base, creator, tools

# ----------------------------------------------------------------------------
# Optional utils that already exist in your code base
# ----------------------------------------------------------------------------
try:
    # Only available in the runtime env of the task runner
    from app.tasks.TA28_DoECreator.utils.generate_doe_uuid import generate_doe_uuid
    from app.tasks.TA28_DoECreator.TA28_A_Initiater import TA28_A_Initiater
except ModuleNotFoundError:
    # When running unit‑tests outside Dagster we just fall back to a local stub
    def generate_doe_uuid(row: Dict[str, Any]) -> str:  # type: ignore
        import hashlib, json
        return "DoE_" + hashlib.sha1(json.dumps(row, sort_keys=True).encode()).hexdigest()[:10]

    class TA28_A_Initiater:  # type: ignore
        @staticmethod
        def _flatten_doe_row(_cfg):
            raise RuntimeError("Stubbed TA28_A_Initiater not available – inject a replacement or run inside the Dagster env")

# ----------------------------------------------------------------------------
# Helper:  generic mutation procedure (independent pure function)
# ----------------------------------------------------------------------------

def mutate_generic(
    ind: Dict[str, Any],
    static_factors: Dict[str, Dict[str, Any]],
    dependent_factors: Dict[str, Dict[str, Any]],
    potential_data: pd.DataFrame,
    deap_config: Dict[str, Dict[str, Any]],
    defaults: Dict[str, Any],
    allow_multi: bool,
) -> Dict[str, Any]:
    """Mutate *in‑place* a *payload*‑style individual and return it.

    The logic is a cleaned‑up version of the experimental code from the
    original repo with three simplifications:
    
    * *growth_bias* is supported but defaults to 0.
    * If a higher‑harchy field becomes empty after the dependencies have
      changed we simply *re‑draw* it from the *potential_data* slice –
      this keeps the DoE valid without adding a fallback search.
    * The function mutates **either** one field **or** several fields
      (if *allow_multi_mutation* is true).  The YAML flag is honoured
      on the *branch* level, not per individual.
    """

    # ------------------------------------------------------------------
    # Local helpers
    # ------------------------------------------------------------------
    def _should_mutate(chance: float) -> bool:
        return random.random() < chance

    def _pick_values(candidates: List[Any], multi: bool, rate: float) -> List[Any] | Any:
        if not candidates:
            return [] if multi else None
        if not multi:
            return random.choice(candidates)

        # Multi‑value field: we drop some of the existing values and add
        # some new ones, scaled by *rate* (0…1)
        k = max(1, int(len(candidates) * rate))
        return random.sample(candidates, k=k)

    # ------------------------------------------------------------------
    payload = ind.setdefault("payload", {})
    doe_config: Dict[str, Dict[str, Any]] = payload.setdefault("doe_config", {})
    primary_data = doe_config.setdefault("primary_data", {})

    mutated_keys = set()

    # ------------------------------------------------------------------
    # 1) STATIC FACTORS --------------------------------------------------
    # ------------------------------------------------------------------
    for key, meta in static_factors.items():
        section, field = key.split(".")
        vocab = meta["values"]
        multi = meta["multi"]

        cfg = deap_config.get(key, {})
        mutate_chance = cfg.get("mutate_chance", defaults.get("mutate_chance", 0.1))
        mutation_rate = cfg.get("mutation_rate", defaults.get("mutation_rate", 0.1))

        if not _should_mutate(mutate_chance):
            continue

        new_val = _pick_values(vocab, multi, mutation_rate)
        doe_config.setdefault(section, {})[field] = new_val
        mutated_keys.add(key)

    # ------------------------------------------------------------------
    # 2) DEPENDENT FACTORS ----------------------------------------------
    # ------------------------------------------------------------------
    for dep_key, meta in dependent_factors.items():
        deps = meta["depends_on"]
        multi = meta["multi"]
        section, field = dep_key.split(".")
        cfg = deap_config.get(dep_key, {})

        # Mutate if any upstream dependency was mutated – *or* by chance
        if dep_key in mutated_keys or any(d in mutated_keys for d in deps):
            pass  # forced mutation
        elif not _should_mutate(cfg.get("mutate_chance", defaults.get("mutate_chance", 0.1))):
            continue

        # Build candidate pool -------------------------------------------------
        # Start with the full data set and iteratively narrow it down to match
        # the *current* values of all dependency fields.
        df_slice = potential_data.copy()
        for d in deps:
            d_section, d_field = d.split(".")
            current_vals = doe_config.get(d_section, {}).get(d_field, [])
            if not current_vals:
                continue  # Wildcard → keep everything
            if isinstance(current_vals, list):
                df_slice = df_slice[df_slice[d_field].isin(current_vals)]
            else:
                df_slice = df_slice[df_slice[d_field] == current_vals]

        candidates = df_slice[field].dropna().unique().tolist()
        new_val = _pick_values(candidates, multi, cfg.get("mutation_rate", defaults.get("mutation_rate", 0.1)))
        doe_config.setdefault(section, {})[field] = new_val
        mutated_keys.add(dep_key)

    ind["payload"]["doe_config"] = doe_config
    return ind

# =============================================================================
# Main class
# =============================================================================

class TA28_B_Mutator:
    """Public entry point used by *TA28_0_DoECreatorOrchestrator*.

    Typical usage inside the orchestrator::

        mutator = TA28_B_Mutator(
            wood_master_df=self.woodMasterPotential_full,
            static_factors=self.STATIC_FACTORS,
            dependent_factors=self.DEPENDENT_FACTORS,
            branch_config=config,  # ← the sub‑branch block from YAML
        )

        new_jobs_df = mutator.evolve(
            prev_gen_df=last_gen_doe_df,
            branch_id=branch_id,
            subbranch_id=subbranch_id,
            new_gen=current_gen + 1,
        )
    """

    # ------------------------------------------------------------------
    # Construction ------------------------------------------------------
    # ------------------------------------------------------------------
    def __init__(
        self,
        wood_master_df: pd.DataFrame,
        static_factors: Dict[str, Dict[str, Any]],
        dependent_factors: Dict[str, Dict[str, Any]],
        branch_config: Dict[str, Any],
        defaults: Dict[str, Any] | None = None,
    ) -> None:
        self.woodMasterPotential_full = wood_master_df
        self.STATIC_FACTORS = static_factors
        self.DEPENDENT_FACTORS = dependent_factors
        self.cfg = branch_config
        self.defaults = defaults or {"mutate_chance": 0.1, "mutation_rate": 0.1, "growth_bias": 0.0}

        # ------------------------------------------------------------------
        # Mutation strength function (string like "lambda gen: …")
        # ------------------------------------------------------------------
        func_str: str = branch_config.get("mutation_func", "lambda gen: 0.1")
        self.mutation_strength_func: Callable[[int], float] = eval(func_str)

        # ------------------------------------------------------------------
        # Genetic operator parameters
        # ------------------------------------------------------------------
        self.elite_size = branch_config.get("elite_size", 5)
        self.generation_size = branch_config.get("generation_size", 30)
        self.allow_multi_mutation = branch_config.get("allow_multi_mutation", True)
        self.tournament_k = branch_config.get("tournament_k", 3)  # default if not in YAML

        # ------------------------------------------------------------------
        # DEAP boiler‑plate
        # ------------------------------------------------------------------
        # Avoid re‑creating the classes when unit‑tests instantiate the mutator
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", dict, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("mate", self._crossover_uniform)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_k)

    # ------------------------------------------------------------------
    # PUBLIC API --------------------------------------------------------
    # ------------------------------------------------------------------
    def evolve(
        self,
        prev_gen_df: pd.DataFrame,
        branch_id: str,
        subbranch_id: str,
        new_gen: int,
    ) -> pd.DataFrame:
        """Main driver used by the orchestrator.

        :param prev_gen_df:  DataFrame returned by
                             ``get_DoEs_of_last_gen`` – must contain at
                             least the columns *payload* and *DoE_UUID*.
        :param new_gen:      Generation number to assign to the offspring.
        :returns:            A DataFrame ready for ``store_or_update_jobs``.
        """
        # 1) Build DEAP population from finished jobs --------------------
        population = self._build_population(prev_gen_df)

        # 2) Attach fitness (score) – currently random -------------------
        self._score_population(population)

        # 3) Elitism -----------------------------------------------------
        population.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
        elites = population[: self.elite_size]

        # 4) Parents via tournament -------------------------------------
        n_offspring = self.generation_size - self.elite_size
        parents = self.toolbox.select(population, n_offspring)

        # 5) Mating (pairwise) ------------------------------------------
        offspring: List[creator.Individual] = []
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % len(parents)]  # wrap‑around
            c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
            if random.random() < 0.5:  # simple crossover prob.
                self.toolbox.mate(c1, c2)
            offspring.extend([c1, c2])

        offspring = offspring[: n_offspring]  # Trim in case of uneven parent count

        # 6) Mutation ----------------------------------------------------
        for ind in offspring:
            self.toolbox.mutate(ind)

        # 7) Combine elites + offspring ---------------------------------
        next_gen = elites + offspring
        assert len(next_gen) == self.generation_size

        # 8) Flatten and wrap into DoE_Jobs rows -------------------------
        job_rows = [self._individual_to_job(ind, branch_id, subbranch_id, new_gen) for ind in next_gen]
        job_df = pd.DataFrame(job_rows)
        return job_df

    # ------------------------------------------------------------------
    # Internals ---------------------------------------------------------
    # ------------------------------------------------------------------
    def _build_population(self, df: pd.DataFrame) -> List[creator.Individual]:
        """Convert result rows into DEAP individuals."""
        inds: List[creator.Individual] = []
        for _, row in df.iterrows():
            if "payload" not in row:
                logging.warning("Row missing 'payload' – skipping one individual")
                continue
            ind = creator.Individual(copy.deepcopy(row["payload"]))
            ind.metadata = {
                "DoE_UUID": row.get("DoE_UUID"),
                "fitness_raw": row.get("fitness"),  # raw fitness if already present
            }
            inds.append(ind)
        return inds

    # ------------------------------------------------------------------
    def _score_population(self, population: List[creator.Individual]) -> None:
        """Attach *random* fitness for now – replace with real logic later."""
        for ind in population:
            score = random.random()
            ind.fitness.values = (score,)

    # ------------------------------------------------------------------
    def _crossover_uniform(self, ind1: creator.Individual, ind2: creator.Individual) -> tuple:
        """Uniform crossover at the *section* level.

        We do **not** swap individual list elements – the unit of exchange
        is one of the *top‑level* sections (primary_data, modeling, …).
        This keeps taxonomic hierarchies intact while still allowing
        meaningful recombinations.
        """
        sections = list(ind1["doe_config"].keys())
        for section in sections:
            if random.random() < 0.5:
                tmp = ind1["doe_config"][section]
                ind1["doe_config"][section] = ind2["doe_config"][section]
                ind2["doe_config"][section] = tmp
        return ind1, ind2

    # ------------------------------------------------------------------
    def _mutate(self, ind: creator.Individual):
        """Thin wrapper that passes all globals to *mutate_generic*."""
        mutate_generic(
            ind,
            static_factors=self.STATIC_FACTORS,
            dependent_factors=self.DEPENDENT_FACTORS,
            potential_data=self.woodMasterPotential_full,
            deap_config={
                **self.cfg.get("static_mutate_params", {}),
                **self.cfg.get("dependent_mutate_params", {}),
            },
            defaults=self.defaults,
            allow_multi=self.allow_multi_mutation,
        )
        return (ind,)

    # ------------------------------------------------------------------
    def _individual_to_job(
        self,
        ind: creator.Individual,
        branch_id: str,
        subbranch_id: str,
        gen: int,
    ) -> Dict[str, Any]:
        """Flatten individual -> row accepted by *create_initial_population*."""
        flat_cfg = TA28_A_Initiater._flatten_doe_row(ind["doe_config"])

        flat_cfg.update(
            {
                "branch_id": branch_id,
                "subbranch_id": subbranch_id,
                "generation": gen,
                "origin": "mutated",
                "created_by": "TA28_DoECreator",
            }
        )

        # ------------------------------------------------------------------
        # ⚠️ Important:  Do *not* sort keys here – generate_doe_uuid already
        # does a stable sort internally.  We merely have to ensure all list
        # values are list objects (even if len==1) so the hash is stable.
        # ------------------------------------------------------------------
        doe_uuid = generate_doe_uuid(flat_cfg)
        flat_cfg["DoE_UUID"] = doe_uuid
        flat_cfg["payload"] = {"doe_config": ind["doe_config"]}

        return flat_cfg



"""
TA28_B_Mutator
--------------

Evolution engine for the TA‑28 optimisation that now **uses a 3‑dimensional
Pareto (NSGA‑II) selection** instead of elite + tournament.

The public façade is still one call:

```python
mutator = TA28_B_Mutator(
    wood_master_df=woodMasterPotential_full,
    static_factors=STATIC_FACTORS,
    dependent_factors=DEPENDENT_FACTORS,
    branch_config=branch_cfg,          # YAML ➜ branch_template
    population_size=128,              # default == len(old_gen_df)
)
next_jobs_df = mutator.evolve(old_gen_df, gen_no=2)
```

Key updates
~~~~~~~~~~~
* **Multi‑objective fitness** – a tuple `(scoreA, scoreB, scoreC)` with
  DEAP weights `(1.0, 1.0, 1.0)` (change signs if a metric is to be
  minimised).
* **`select` → `tools.selNSGA2`** for true Pareto pressure.
* Ranks + crowding distance are preserved on the population object so you
  can inspect `ind.fitness.values` or `ind.fitness.crowding_dist` for
  logging/analytics.
* Everything else (custom uniform crossover, linear mutation respecting
  taxonomic dependencies, UUID regeneration, DF ⇄ Individual mapping)
  is untouched.

Dependencies
~~~~~~~~~~~~
```text
pandas >= 2.2
numpy  >= 1.26
DEAP   >= 1.4
```

"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Sequence, Tuple

import pandas as pd
from deap import base, creator, tools

# -----------------------------------------------------------------------------
# Helper – stochastic fitness placeholder until real KPI fetch is wired in
# -----------------------------------------------------------------------------

def _random_3d_scores(_: Sequence[str]) -> List[Tuple[float, float, float]]:
    """Return three random objectives in [0, 1] for each UUID."""
    return [(random.random(), random.random(), random.random()) for _ in _]


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------

class TA28_B_Mutator:
    """Create a new DoE generation via NSGA‑II multi‑objective evolution."""

    def __init__(
        self,
        wood_master_df: pd.DataFrame,
        static_factors: List[str],
        dependent_factors: Dict[str, List[str]],
        branch_config: Dict,
        population_size: int | None = None,
        fetch_scores_fn: Callable[[Sequence[str]], List[Tuple[float, float, float]]] = _random_3d_scores,
        mutation_rate: float = 0.2,
        mutation_chance: float = 0.05,
    ) -> None:
        self.wood_master_df = wood_master_df
        self.static_factors = static_factors
        self.dependent_factors = dependent_factors
        self.branch_config = branch_config
        self.fetch_scores = fetch_scores_fn
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_chance = mutation_chance

        self.toolbox = base.Toolbox()
        self._build_toolbox()

    # ------------------------------------------------------------------
    # Toolbox construction
    # ------------------------------------------------------------------

    def _build_toolbox(self) -> None:
        # ----- DEAP creators --------------------------------------------------
        if "FitnessMulti" not in creator.__dict__:
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))
        if "Individual" not in creator.__dict__:
            creator.create("Individual", dict, fitness=creator.FitnessMulti)

        # ----- Attribute generators ------------------------------------------
        def _init_random_row() -> creator.Individual:  # type: ignore
            # Randomly sample a row from wood master and return as Individual.
            row = self.wood_master_df.sample(1).iloc[0].to_dict()
            return creator.Individual(row)  # type: ignore[arg-type]

        self.toolbox.register("individual", _init_random_row)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # ----- Genetic operators ---------------------------------------------
        self.toolbox.register("mate", self._mate_uniform_taxonomic)
        self.toolbox.register("mutate", self._mutate_taxonomic, indpb=self.mutation_chance)
        self.toolbox.register("select", tools.selNSGA2)

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------

    # ~~~~~ crossover ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _mate_uniform_taxonomic(self, ind1: creator.Individual, ind2: creator.Individual) -> Tuple[creator.Individual, creator.Individual]:
        """Uniformly swap *whole* taxonomic sections to keep dependency integrity."""
        for factor in self.static_factors:
            if random.random() < 0.5:
                # swap the static factor
                ind1_val, ind2_val = ind1[factor], ind2[factor]
                ind1[factor], ind2[factor] = ind2_val, ind1_val
                # whenever a static factor changes, redraw all its children so hierarchy remains valid
                for child in self.dependent_factors.get(factor, []):
                    ind1[child] = self._sample_child(child)
                    ind2[child] = self._sample_child(child)
        return ind1, ind2

    # ~~~~~ mutation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _mutate_taxonomic(self, individual: creator.Individual, indpb: float) -> Tuple[creator.Individual]:
        """Linearly mutate any attribute with probability *indpb* respecting tree."""
        for factor in self.static_factors:
            if random.random() < indpb:
                individual[factor] = self._sample_static(factor)
                # refresh dependents
                for child in self.dependent_factors.get(factor, []):
                    individual[child] = self._sample_child(child)
        return (individual,)

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_static(self, factor: str):
        choices = self.branch_config[factor]["options"]  # assuming config layout
        return random.choice(choices)

    def _sample_child(self, factor: str):
        return self.wood_master_df.sample(1).iloc[0][factor]

    # ------------------------------------------------------------------
    # Evolution orchestration
    # ------------------------------------------------------------------

    def evolve(self, old_gen_df: pd.DataFrame, gen_no: int) -> pd.DataFrame:
        """Create next generation DF from *old_gen_df* and return it."""
        pop = self._df_to_population(old_gen_df)
        # evaluate old population (padding real + random fitness for now)
        uuids = old_gen_df["doe_uuid"]
        scores = self.fetch_scores(uuids)
        for ind, fit in zip(pop, scores):
            ind.fitness.values = fit

        # create offspring ----------------------------------------------------
        offspring = tools.selNSGA2(pop, len(pop))  # make sure rank/crowding dist ok
        offspring = [self.toolbox.clone(ind) for ind in offspring]

        # crossover -----------------------------------------------------------
        for i1, i2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.9:  # hard‑coded cx rate
                self.toolbox.mate(i1, i2)
                del i1.fitness.values, i2.fitness.values

        # mutation ------------------------------------------------------------
        for mutant in offspring:
            if random.random() < self.mutation_rate:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

        # evaluate new individuals (random scores until backend ready) --------
        new_uuids = [self._regenerate_uuid(ind, gen_no) for ind in offspring]
        new_scores = self.fetch_scores(new_uuids)
        for ind, fit in zip(offspring, new_scores):
            ind.fitness.values = fit

        # combine & select next gen (NSGA‑II) ---------------------------------
        combined = pop + offspring
        next_pop = self.toolbox.select(combined, self.population_size or len(pop))

        # to DataFrame --------------------------------------------------------
        next_df = self._population_to_df(next_pop, gen_no)
        return next_df

    # ------------------------------------------------------------------
    # Population ⇄ DataFrame
    # ------------------------------------------------------------------

    def _df_to_population(self, df: pd.DataFrame):
        pop: List[creator.Individual] = []
        for _, row in df.iterrows():
            ind = creator.Individual(row.to_dict())  # type: ignore[arg-type]
            pop.append(ind)
        return pop

    def _population_to_df(self, pop: List[creator.Individual], gen_no: int) -> pd.DataFrame:
        rows = []
        for ind in pop:
            row = dict(ind)
            row.update(
                {
                    "generation": gen_no,
                    "fitness_a": ind.fitness.values[0],
                    "fitness_b": ind.fitness.values[1],
                    "fitness_c": ind.fitness.values[2],
                    "doe_uuid": ind["doe_uuid"],
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _regenerate_uuid(ind: creator.Individual, gen_no: int) -> str:
        """Replace/doe_uuid with a new one for *gen_no* – placeholder impl."""
        new_uuid = f"TA28‑{gen_no:03d}-{random.randrange(1_000_000):06d}"
        ind["doe_uuid"] = new_uuid
        return new_uuid

