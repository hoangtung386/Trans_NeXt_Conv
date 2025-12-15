"""Evaluation Script for Multi-Organ Segmentation Model"""
import os
import torch
from configs.config import CONFIG
from src.models.initialize_model import get_model
from src.data.dataloader import get_loaders
from src.evaluation.evaluator import SegmentationEvaluator

if __name__ == "__main__":
    # Ensure output directory exists
    output_dir = 'Trans_next_Conv/images/evaluation_results'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model
    model = get_model(CONFIG)
    
    # Get loaders
    _, val_loader = get_loaders(CONFIG)

    # Initialize Evaluator
    evaluator = SegmentationEvaluator(
        model, 
        val_loader, 
        CONFIG['device'], 
        num_classes=CONFIG['num_classes'],
        output_dir=output_dir
    )

    # Compute metrics
    print("Computing metrics")
    results = evaluator.compute_metrics()
    
    # Plot metrics comparison
    print("Plotting metrics comparison")
    evaluator.plot_metrics(results)
    
    # Visualize overlay masks
    print("Creating overlay visualizations")
    evaluator.visualize_predictions(num_samples=5)

    # Per-class comparison
    print("Creating per-class comparison")
    evaluator.plot_per_class_comparison(num_samples=3)
    
    # Confusion matrix analysis
    print("Creating confusion matrix")
    evaluator.plot_confusion_analysis()
    
    # Create summary report
    evaluator.create_summary_report(results)
