"""Evaluation Script for Multi-Organ Segmentation Model"""
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.colors import ListedColormap
from tqdm import tqdm

from configs.config import CONFIG
from src.models.initialize_model import get_model
from src.data.dataloader import get_loaders

output_dir = 'Trans_next_Conv/images/evaluation_results'
os.makedirs(output_dir, exist_ok=True)

class SegmentationEvaluator:
    def __init__(self, model, val_loader, device, num_classes=6):
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        self.class_names = ['Background', 'Liver', 'Spleen', 'Kidney_L', 'Kidney_R', 'Bowel']

    def compute_metrics(self):
        """Compute comprehensive metrics"""
        self.model.eval()

        class_metrics = {cls: {
            'dice': [], 'iou': [], 'precision': [], 
            'recall': [], 'specificity': []
        } for cls in range(1, self.num_classes)}

        print("Computing metrics on validation set...")
        with torch.no_grad():
            for batch in tqdm(self.val_loader):
                images = batch['image'].to(self.device)
                labels = batch['seg'].to(self.device)

                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)

                pred_np = preds.cpu().numpy()
                label_np = labels.cpu().numpy()

                for cls in range(1, self.num_classes):
                    pred_cls = (pred_np == cls).astype(np.float32)
                    label_cls = (label_np == cls).astype(np.float32)

                    if label_cls.sum() == 0:
                        continue

                    intersection = (pred_cls * label_cls).sum()
                    dice = (2 * intersection) / (pred_cls.sum() + label_cls.sum() + 1e-8)
                    class_metrics[cls]['dice'].append(dice)

                    union = pred_cls.sum() + label_cls.sum() - intersection
                    iou = intersection / (union + 1e-8)
                    class_metrics[cls]['iou'].append(iou)

                    tp = intersection
                    fp = pred_cls.sum() - intersection
                    fn = label_cls.sum() - intersection
                    tn = ((pred_cls == 0) & (label_cls == 0)).sum()

                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    specificity = tn / (tn + fp + 1e-8)

                    class_metrics[cls]['precision'].append(precision)
                    class_metrics[cls]['recall'].append(recall)
                    class_metrics[cls]['specificity'].append(specificity)

        results = {}
        for cls in range(1, self.num_classes):
            results[self.class_names[cls]] = {
                metric: np.mean(values) if values else 0.0
                for metric, values in class_metrics[cls].items()
            }

        return results

    def plot_metrics(self, results):
        """Plot metrics comparison"""
        metrics_to_plot = ['dice', 'iou', 'precision', 'recall', 'specificity']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_plot):
            classes = list(results.keys())
            values = [results[cls][metric] for cls in classes]

            axes[idx].bar(classes, values, color='steelblue', alpha=0.8)
            axes[idx].set_title(f'{metric.upper()}', fontsize=14, fontweight='bold')
            axes[idx].set_ylabel('Score', fontsize=12)
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(axis='y', alpha=0.3)
            axes[idx].tick_params(axis='x', rotation=45)

            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)

        axes[5].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_comparison_2d.png'), dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Metrics plot saved to {output_dir}/metrics_comparison_2d.png")

    def visualize_predictions(self, num_samples=5):
        """Visualize overlay masks for 2D slices"""
        self.model.eval()

        colors = ['black', 'red', 'green', 'blue', 'yellow', 'purple']
        cmap = ListedColormap(colors)

        samples_shown = 0

        with torch.no_grad():
            for batch in self.val_loader:
                if samples_shown >= num_samples:
                    break

                images = batch['image'].to(self.device)
                labels = batch['seg'].to(self.device)

                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)

                img_np = images[0, 0].cpu().numpy()
                pred_np = preds[0].cpu().numpy()
                label_np = labels[0, 0].cpu().numpy()

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Original image
                axes[0].imshow(img_np, cmap='gray')
                axes[0].set_title('CT Image', fontsize=14, fontweight='bold')
                axes[0].axis('off')

                # Ground truth
                axes[1].imshow(img_np, cmap='gray')
                axes[1].imshow(label_np, cmap=cmap, alpha=0.5, vmin=0, vmax=5)
                axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
                axes[1].axis('off')

                # Prediction
                axes[2].imshow(img_np, cmap='gray')
                axes[2].imshow(pred_np, cmap=cmap, alpha=0.5, vmin=0, vmax=5)
                axes[2].set_title('Prediction', fontsize=14, fontweight='bold')
                axes[2].axis('off')

                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor=colors[i], label=self.class_names[i])
                                 for i in range(1, len(colors))]
                fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=12)

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'overlay_2d_sample_{samples_shown+1}.png'),
                           dpi=150, bbox_inches='tight')
                plt.show()

                samples_shown += 1

        print(f"Overlay visualizations saved to {output_dir}/overlay_2d_sample_*.png")

    def plot_per_class_comparison(self, num_samples=3):
        """Plot detailed per-class segmentation comparison"""
        self.model.eval()
        
        colors = ['black', 'red', 'green', 'blue', 'yellow', 'purple']
        
        samples_shown = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if samples_shown >= num_samples:
                    break
                
                images = batch['image'].to(self.device)
                labels = batch['seg'].to(self.device)
                
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                
                img_np = images[0, 0].cpu().numpy()
                pred_np = preds[0].cpu().numpy()
                label_np = labels[0, 0].cpu().numpy()
                
                # Create figure with per-organ comparison
                fig, axes = plt.subplots(2, self.num_classes, figsize=(20, 8))
                
                for cls_idx in range(self.num_classes):
                    # Ground truth for this class
                    gt_mask = (label_np == cls_idx).astype(float)
                    axes[0, cls_idx].imshow(img_np, cmap='gray')
                    axes[0, cls_idx].imshow(gt_mask, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
                    axes[0, cls_idx].set_title(f'{self.class_names[cls_idx]}\nGround Truth', 
                                               fontsize=10, fontweight='bold')
                    axes[0, cls_idx].axis('off')
                    
                    # Prediction for this class
                    pred_mask = (pred_np == cls_idx).astype(float)
                    axes[1, cls_idx].imshow(img_np, cmap='gray')
                    axes[1, cls_idx].imshow(pred_mask, cmap='Blues', alpha=0.5, vmin=0, vmax=1)
                    axes[1, cls_idx].set_title(f'{self.class_names[cls_idx]}\nPrediction', 
                                               fontsize=10, fontweight='bold')
                    axes[1, cls_idx].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'per_class_comparison_{samples_shown+1}.png'),
                           dpi=150, bbox_inches='tight')
                plt.show()
                
                samples_shown += 1
        
        print(f"Per-class comparison saved to {output_dir}/per_class_comparison_*.png")

    def plot_confusion_analysis(self):
        """Plot confusion matrix for segmentation classes"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        print("Computing confusion matrix...")
        with torch.no_grad():
            for batch in tqdm(self.val_loader):
                images = batch['image'].to(self.device)
                labels = batch['seg'].to(self.device)
                
                outputs = self.model(images)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.append(preds.cpu().numpy().flatten())
                all_labels.append(labels.cpu().numpy().flatten())
        
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=range(self.num_classes))
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    ax=ax1, cbar_kws={'label': 'Count'})
        ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        
        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Greens',
                    xticklabels=self.class_names, yticklabels=self.class_names,
                    ax=ax2, cbar_kws={'label': 'Proportion'})
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.set_xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_2d.png'), dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Confusion matrix saved to {output_dir}/confusion_matrix_2d.png")

    def create_summary_report(self, results):
        """Create text summary report"""
        report_path = os.path.join(output_dir, 'evaluation_report_2d.txt')

        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("2D SEGMENTATION EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")

            for cls_name, metrics in results.items():
                f.write(f"\n{cls_name}:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Dice Coefficient:  {metrics['dice']:.4f}\n")
                f.write(f"  IoU:              {metrics['iou']:.4f}\n")
                f.write(f"  Precision:        {metrics['precision']:.4f}\n")
                f.write(f"  Recall:           {metrics['recall']:.4f}\n")
                f.write(f"  Specificity:      {metrics['specificity']:.4f}\n")

            f.write("\n" + "-"*60 + "\n")
            f.write("OVERALL AVERAGES:\n")
            f.write("-"*60 + "\n")

            for metric in ['dice', 'iou', 'precision', 'recall', 'specificity']:
                avg = np.mean([results[cls][metric] for cls in results.keys()])
                f.write(f"  Average {metric.upper()}: {avg:.4f}\n")

        print(f"\nEvaluation report saved to {report_path}")

        with open(report_path, 'r') as f:
            print(f.read())


if __name__ == "__main__":
    model = get_model(CONFIG)
    # Start from scratch or load weights? Usually evaluate needs weights.
    # Assuming user handles weight loading if running this script directly.
    # For now just instantiating structure.
    
    _, val_loader = get_loaders(CONFIG)

    evaluator = SegmentationEvaluator(model, val_loader, 
                    CONFIG.device, num_classes=CONFIG['num_classes'])

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
