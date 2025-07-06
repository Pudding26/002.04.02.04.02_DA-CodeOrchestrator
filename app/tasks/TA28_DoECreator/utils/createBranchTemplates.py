from pathlib import Path
import yaml

def generate_branch_config():
    def linear_func(id_num): return f"lambda gen: max(0.05, min(0.45, {id_num:.3f} * gen / 50))"
    def inverse_func(id_num): return f"lambda gen: max(0.05, min(0.45, {id_num:.3f} / (gen + 1)))"
    def random_func(_): return "lambda gen: random.uniform(0.05, 0.45)"
    def constant_func(id_num): return f"lambda gen: {0.05 + 0.1 * (id_num - 1):.3f}"

    base_config = {
        "allow_multi_mutation": True,
        "fallback_on_invalid": True,
        "corridor": 0.1,
        "max_generations": 50,
        "generation_size": 30,
        "elite_size": 5,
        "constraints": {
            "include": {
                "primary_data.sourceNo": ["DS01", "DS04"],
                "primary_data.woodType": ["Softwood"]
            },
            "exclude": {
                "primary_data.genus": ["Quercus", "Eucalyptus"]
            }
        }
    }

    config = {}
    control_count = 5

    # Define strategies
    strategies = [
        ("linear", linear_func),
        ("inverse", inverse_func),
        ("random", random_func),
        ("constant", constant_func)
    ]

    # Generate 5 branches per strategy
    for strategy, func_generator in strategies:
        for i in range(1, 6):
            branch_id = f"branch_{strategy}_{i:02d}"
            param_func = func_generator(i)

            config[branch_id] = {
                "mutation_strategy": strategy,
                "mutation_func": param_func,
                "control_branches": [f"{branch_id}_{chr(65 + j)}" for j in range(control_count)],
                **base_config
            }

    output_path = Path("generated_branch_config.yaml")
    with open(output_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    return output_path

generate_branch_config()
