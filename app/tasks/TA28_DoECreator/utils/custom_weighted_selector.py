import math
import random
import pandas as pd
from deap.tools import sortNondominated

def custom_weighted_selector(population, k):
    """
    Custom Pareto-based selector with audit summary.

    - Sort individuals into Pareto fronts
    - Assign weight by front index (weight halves with each successive front)
    - Return selected individuals and audit DataFrame

    Args:
        population: list of DEAP individuals
        k: number of individuals to select

    Returns:
        selected: list of selected individuals
        df_summary: pandas DataFrame with audit information
    """

    fronts = sortNondominated(population, len(population), first_front_only=False)

    records = []
    individuals = []
    weights = []

    for front_idx, front in enumerate(fronts):
        front_no = front_idx + 1
        fw = 0.5 ** front_no  # halve per front: 0.5, 0.25, 0.125, 0.0625, ...

        for ind in front:
            doe_uuid = getattr(ind, "doe_uuid", None)
            individuals.append(ind)
            weights.append(fw)

            record = {
                "DoE_UUID": doe_uuid,
                "front": front_no,
                "weight": fw
            }

            for obj_idx, obj_val in enumerate(ind.fitness.values, start=1):
                record[f"Objective {obj_idx} value"] = obj_val

            records.append(record)

    total_weight = sum(weights)
    probs = [w / total_weight for w in weights]

    selected = random.choices(individuals, weights=probs, k=k)



    df_summary = pd.DataFrame(records)

    return selected, df_summary
