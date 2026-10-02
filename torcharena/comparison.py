from torcharena.config import RunConfig
from torcharena.safety import TrainingError
from torcharena.service import train


def fairness(config: dict) -> dict:
    return {
        "dataset": config["dataset"],
        "seed": config["experiment"]["seed"],
        "training": {
            key: config["training"][key]
            for key in ("epochs", "batch_size", "device", "amp", "deterministic")
        },
        "early_stopping": config["early_stopping"],
    }


def compare(records: list[dict]) -> dict:
    if len(records) < 2:
        raise ValueError("Comparison needs at least two runs")
    comparable = all(fairness(r["config"]) == fairness(records[0]["config"]) for r in records)
    completed = all(r["status"] == "COMPLETED" for r in records)
    rows = []
    for record in records:
        summary = record["summary"]
        metrics = summary.get("final_metrics", {})
        rows.append(
            {
                "run_id": record["run_id"],
                "model": record["config"]["model"]["name"],
                "status": record["status"],
                "dataset": record["config"]["dataset"]["name"],
                "seed": record["config"]["experiment"]["seed"],
                "epochs": summary.get("last_epoch"),
                "accuracy_fraction": metrics.get("val_accuracy"),
                "val_loss": metrics.get("val_loss"),
                "seconds": summary.get("duration_seconds"),
                "parameters": summary.get("parameters"),
            }
        )
    categories = {}
    if comparable and completed:
        for label, key, reverse in [
            ("Highest accuracy", "accuracy_fraction", True),
            ("Lowest validation loss", "val_loss", False),
            ("Fastest measured training", "seconds", False),
            ("Fewest parameters", "parameters", False),
        ]:
            if all(row[key] is not None for row in rows):
                best = (max if reverse else min)(row[key] for row in rows)
                categories[label] = [row["run_id"] for row in rows if row[key] == best]
    return {
        "comparable": comparable,
        "complete": completed,
        "rows": rows,
        "categories": categories,
        "note": "COMPARABLE" if comparable else "NOT COMPARABLE: dataset/seed/budget/device differ",
    }


def race(repository, configs: list[RunConfig]) -> dict:
    if len(configs) < 2:
        raise ValueError("Race requires at least two configs")
    if any(fairness(c.model_dump()) != fairness(configs[0].model_dump()) for c in configs[1:]):
        raise ValueError("NOT COMPARABLE: race requires identical dataset, seed, budget and device")
    run_ids = []
    for config in configs:
        run_id = train(repository, config)
        run_ids.append(run_id)
        if repository.get(run_id)["status"] == "INTERRUPTED":
            repository.save_race({"run_ids": run_ids, "status": "INTERRUPTED"})
            raise TrainingError(
                "race_interrupted",
                f"Race stopped; resume contestant with torcharena resume {run_id}",
            )
    result = compare([repository.get(run_id) for run_id in run_ids])
    result["race_id"] = repository.save_race(result)
    return result
