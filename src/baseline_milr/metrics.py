import csv
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record) + "\n")


def _score(reward, data_name):
    return float(reward + 1.0 if data_name == "geneval" else reward)


def _steps_to_threshold(scores, reference_score, fraction):
    if reference_score is None:
        return None
    threshold = fraction * reference_score
    for step, score in enumerate(scores, start=1):
        if score >= threshold:
            return step
    return None


def build_example_metrics(
    example,
    source_index,
    reward_history,
    trajectory,
    budget,
    data_name,
    milr_16_score,
):
    scores = [_score(reward, data_name) for reward in reward_history]
    step_scores = scores[1:] if len(scores) > 1 else [scores[0]]
    step_scores = step_scores[:budget]
    step_scores.extend([step_scores[-1]] * (budget - len(step_scores)))

    output_trajectory = []
    for item in trajectory:
        record = dict(item)
        record["score"] = _score(item["reward"], data_name)
        record["best_score"] = _score(item["best_reward"], data_name)
        output_trajectory.append(record)

    return {
        "source_index": source_index,
        "tag": example["tag"],
        "prompt": example["prompt"],
        "initial_reward": float(reward_history[0]),
        "final_reward": float(reward_history[-1]),
        "best_reward": float(max(reward_history)),
        "initial_score": scores[0],
        "final_score": scores[-1],
        "best_score": max(scores),
        "score_by_step": step_scores,
        "anytime_auc": float(np.mean(step_scores)),
        "milr_16_reference_score": milr_16_score,
        "steps_to_95_percent_milr16": _steps_to_threshold(
            step_scores, milr_16_score, 0.95
        ),
        "steps_to_99_percent_milr16": _steps_to_threshold(
            step_scores, milr_16_score, 0.99
        ),
        "executed_steps": len(reward_history) - 1,
        "drift_cost": float(sum(item["drift_cost"] for item in trajectory[1:])),
        "trajectory": output_trajectory,
    }


def plot_trajectory(trajectory, path, title):
    steps = [item["step"] for item in trajectory]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(steps, [item["score"] for item in trajectory], marker="o", label="Score")
    axes[0].plot(
        steps,
        [item["best_score"] for item in trajectory],
        marker="o",
        label="Best",
    )
    axes[0].set_title("Score trajectory")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()
    axes[1].plot(
        steps,
        [item["drift_cost"] for item in trajectory],
        marker="o",
    )
    axes[1].set_title("Latent drift")
    for axis in axes:
        axis.set_xlabel("Refinement step")
        axis.grid(alpha=0.25)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _category_metrics(items):
    score_matrix = np.asarray([item["score_by_step"] for item in items], dtype=float)
    mean_scores = score_matrix.mean(axis=0)
    return {
        "num_examples": len(items),
        "mean_score_by_step": [
            {"step": step, "score": float(score)}
            for step, score in enumerate(mean_scores, start=1)
        ],
        "anytime_auc": float(mean_scores.mean()),
        "mean_initial_score": float(np.mean([item["initial_score"] for item in items])),
        "mean_final_score": float(np.mean([item["final_score"] for item in items])),
        "mean_best_score": float(np.mean([item["best_score"] for item in items])),
        "mean_drift_cost": float(np.mean([item["drift_cost"] for item in items])),
    }


def aggregate_metrics(summaries, budget, milr_16_score, output_dir):
    score_matrix = np.asarray([item["score_by_step"] for item in summaries], dtype=float)
    mean_scores = score_matrix.mean(axis=0)
    initial_scores = np.asarray([item["initial_score"] for item in summaries], dtype=float)
    best_scores = np.maximum.accumulate(
        np.column_stack((initial_scores, score_matrix)), axis=1
    ).mean(axis=0)

    mean_drift_by_step = []
    for step in range(budget + 1):
        values = [
            item["trajectory"][step]["drift_cost"]
            for item in summaries
            if step < len(item["trajectory"])
        ]
        mean_drift_by_step.append(float(sum(values) / len(summaries)))

    reference_score = milr_16_score
    if reference_score is None and budget == 16:
        reference_score = float(mean_scores[-1])

    aggregate = {
        "split": "test",
        "num_examples": len(summaries),
        "mean_score_by_step": [
            {"step": step, "score": float(score)}
            for step, score in enumerate(mean_scores, start=1)
        ],
        "anytime_auc": float(mean_scores.mean()),
        "milr_16_reference_score": reference_score,
        "steps_to_95_percent_milr16": _steps_to_threshold(
            mean_scores, reference_score, 0.95
        ),
        "steps_to_99_percent_milr16": _steps_to_threshold(
            mean_scores, reference_score, 0.99
        ),
        "mean_initial_score": float(initial_scores.mean()),
        "mean_final_score": float(np.mean([item["final_score"] for item in summaries])),
        "mean_best_score": float(np.mean([item["best_score"] for item in summaries])),
        "mean_executed_steps": float(
            np.mean([item["executed_steps"] for item in summaries])
        ),
        "mean_drift_cost": float(np.mean([item["drift_cost"] for item in summaries])),
    }

    categories = defaultdict(list)
    for item in summaries:
        categories[item["tag"]].append(item)
    aggregate["categories"] = {
        tag: _category_metrics(items) for tag, items in sorted(categories.items())
    }

    plot_rows = [
        {
            "step": step,
            "mean_score": float(initial_scores.mean()) if step == 0 else float(mean_scores[step - 1]),
            "mean_best_score": float(best_scores[step]),
            "mean_drift_cost": mean_drift_by_step[step],
        }
        for step in range(budget + 1)
    ]
    aggregate["mean_trajectory_by_step"] = plot_rows
    write_json(os.path.join(output_dir, "metrics.json"), aggregate)
    with open(os.path.join(output_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=plot_rows[0].keys())
        writer.writeheader()
        writer.writerows(plot_rows)

    plot_trajectory(
        [
            {
                "step": row["step"],
                "score": row["mean_score"],
                "best_score": row["mean_best_score"],
                "drift_cost": row["mean_drift_cost"],
            }
            for row in plot_rows
        ],
        os.path.join(output_dir, "metrics.png"),
        "Baseline MILR test",
    )
    return aggregate
