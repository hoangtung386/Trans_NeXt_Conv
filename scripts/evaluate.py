"""Evaluation script for multi-organ segmentation model."""

import os

from configs.config import CONFIG
from src.data.dataloader import get_loaders
from src.evaluation.evaluator import SegmentationEvaluator
from src.models.initialize_model import get_model

if __name__ == "__main__":
    output_dir = "Trans_next_Conv/images/evaluation_results"
    os.makedirs(output_dir, exist_ok=True)

    model = get_model(CONFIG)
    _, val_loader = get_loaders(CONFIG)

    evaluator = SegmentationEvaluator(
        model,
        val_loader,
        CONFIG["device"],
        num_classes=CONFIG["num_classes"],
        output_dir=output_dir,
    )

    print("Computing metrics")
    results = evaluator.compute_metrics()

    print("Plotting metrics comparison")
    evaluator.plot_metrics(results)

    print("Creating overlay visualizations")
    evaluator.visualize_predictions(num_samples=5)

    print("Creating per-class comparison")
    evaluator.plot_per_class_comparison(num_samples=3)

    print("Creating confusion matrix")
    evaluator.plot_confusion_analysis()

    evaluator.create_summary_report(results)
