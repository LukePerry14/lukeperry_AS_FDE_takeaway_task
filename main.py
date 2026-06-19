from __future__ import annotations

import datetime
import json
from src.model_handling import PersonaLLM
from src.societies import Society, DemoAttr
from itertools import product
from pathlib import Path
import random
import argparse
import datetime
import numpy as np

random.seed(42)

INCOME = ["Less than $30,000", "$30,000-$50,000", "$50,000-$75,000", "$75,000-$100,000", "$100,000 or more"]
AGE = ["18-29", "30-49", "50-64", "65+"]
RACE = ["White", "Black", "Hispanic", "Asian", "Other"]


def generate_personas(pop_size=None):
    """One persona per demographic cell: 5 incomes * 4 ages * 5 races = 100 personas."""
    personas = []

    for i, (income, age, race) in enumerate(product(INCOME, AGE, RACE)):
        personas.append(DemoAttr(id=i, income=income, age=age, race=race))

    if pop_size:
        persona_subset = random.sample(personas, pop_size)
        for idx, persona in enumerate(persona_subset):
            persona.id=idx

        return persona_subset


    return personas


def main(qs, strategies, pop_size=None, tranches=False):
    # Extract Question subset from wave54 (eval set)
    try:
        question_arr = json.load(open("src/w54_selected.json"))
        q_dict = {q["qkey"]: q for q in question_arr if q["qkey"] in qs}
    except IOError as e:
        print(f"Failed to open chosen question subset: {e}")
        return

    # Build Society and LLM
    personas = generate_personas(pop_size)
    soc = Society(personas, q_dict)

    # Pre-generate backstories once if any strategy needs them
    
    if any(s in strategies for s in ("backstory", "backstory_socialised")):
        print("Generating Persona Backstories (this will take a long time)...")
        soc.build_backstories(social="backstory_socialised" in strategies)

    print("Finished Generating")

    print("Spinning up generator LLM...")
    llm = PersonaLLM()

    from scipy.stats import wasserstein_distance as wd_fn
    results = {}
    for qkey, q in q_dict.items():
        options = q["options"][:-1]
        ordinal = np.array(q["ordinal"], dtype=float)
        uniform = [1 / len(options)] * len(options)
        wd_uniform = wd_fn(ordinal, ordinal, q["responses"], uniform) / (ordinal.max() - ordinal.min())
        results[qkey] = {
            "question": q["question"],
            "options": options,
            "ground_truth": q["responses"],
            "uniform_upper_bound": {"wd": wd_uniform, "distribution": uniform},
        }

    # Run each strategy
    for strategy in strategies:
        print(f"Strategy: {strategy}")
        for qkey, q in q_dict.items():
            options = q["options"][:-1]

            def _ask(person, neib, _q=q, _opts=options):
                if strategy == "baseline":
                    return llm.ask(person, _q["question"], _opts)
                elif strategy == "socialised":
                    return llm.ask(person, _q["question"], _opts, neighbours=neib)
                elif strategy == "backstory":
                    return llm.ask(person, _q["question"], _opts, backstory=person.backstory)
                elif strategy == "backstory_socialised":
                    return llm.ask(person, _q["question"], _opts, backstory=person.social_backstory)
                else:
                    raise Exception("Invalid Strategy or preprocessing")

            responses = []
            for idx, person in enumerate(soc.people):
                neighbours = np.random.choice(soc.people, 2, p=soc.adj[idx])
                responses.append(_ask(person, neighbours))

            soc.record(qkey, responses)
            wd, dist = soc.evaluate(qkey, tranches=tranches)
            results[qkey][strategy] = {"wd": wd, "distribution": dist}
            print(f"[{strategy}] {qkey}: WD = {wd:.4f}")

    Path("results").mkdir(exist_ok=True)
    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    with open(f"results/{run_name}", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to results/{run_name}")


if __name__ == "__main__":
    # ap = argparse.ArgumentParser()

    # ap.add_argument(
    #     "--qs",
    #     nargs="+",
    #     default=["INEQ5_i_W54", "INEQ8_c_W54", "ECON5_d_W54"],
    # )

    # ap.add_argument(
    #     "--strategies",
    #     nargs="+",
    #     default=["baseline", "socialised", "backstory", "backstory_socialised"],
    # )

    # ap.add_argument(
    #     "--pop_size",
    #     default=None
    # )

    # ap.add_argument(
    #     "--tranches",
    #     default=False,
    #     action="store_true"
    # )

    # args = ap.parse_args()

    # print(
    #     "Running a survey simulation with:\n",
    #     f"Questions: {args.qs}\n",
    #     f"Strategies: {args.strategies}\n",
    #     f"Population Subset size: {f'{args.pop_size}/100' if args.pop_size else '100/100'}\n",
    #     f"Personas as Population Tranches: {'True' if args.tranches else 'False'}\n"
    #     )

    args = argparse.Namespace(
        qs=["INEQ5_i_W54", "INEQ8_c_W54", "ECON5_d_W54"],
        strategies=["backstory_socialised"],  # ["baseline", "socialised", "backstory", "backstory_socialised"],
        pop_size=10,
        tranches=False,
    )

    main(qs=args.qs, strategies=args.strategies, tranches=args.tranches, pop_size=args.pop_size)