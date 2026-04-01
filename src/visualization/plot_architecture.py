"""Architecture visualization using torchviz and torchview."""

import gc
import os

import torch
import torch.nn as nn
from torch.amp import autocast
from torchinfo import summary as torchinfo_summary

from configs.config import CONFIG
from src.models.initialize_model import get_model


class AutocastWrapper(nn.Module):
    """Wrapper to enable mixed precision during visualization."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        with autocast("cuda"):
            return self.model(x)


def generate_model_summary(model, device):
    """Generate model summary using torchinfo."""
    gc.collect()
    torch.cuda.empty_cache()

    dummy_input = torch.randn(1, 1, 256, 256).to(device).half()
    model_wrapped = AutocastWrapper(model)

    print("\n" + "=" * 60)
    print("FP16 (MIXED PRECISION) MODEL SUMMARY")
    print("=" * 60)

    stats = torchinfo_summary(
        model_wrapped,
        input_data=dummy_input,
        col_names=[
            "input_size", "output_size", "num_params",
            "kernel_size", "mult_adds",
        ],
        depth=4,
        device=device,
        verbose=1,
    )

    params_size = stats.total_param_bytes / (1024 ** 2)
    print(f"\nTotal params size (Mixed Precision): {params_size:.2f} MB")

    return stats


def generate_computational_graph(model, device, output_dir="."):
    """Generate computational graph using torchviz."""
    from torchviz import make_dot

    gc.collect()
    torch.cuda.empty_cache()

    dummy_input = torch.randn(1, 1, 256, 256).to(device)
    model.eval()

    with autocast("cuda"):
        output = model(dummy_input)[0]

    print("Generating computational graph visualization...")

    dot = make_dot(
        output,
        params=dict(model.named_parameters()),
        show_attrs=True,
        show_saved=False,
    )

    dot.attr(rankdir="TB")
    dot.attr(
        "node", shape="box", style="rounded,filled", fillcolor="lightblue",
    )
    dot.attr("graph", size="20,20", dpi="300")

    for fmt in ["png", "svg", "pdf"]:
        dot.format = fmt
        filepath = os.path.join(output_dir, "model_architecture_full")
        dot.render(filepath, cleanup=True)
        print(f"Saved: {filepath}.{fmt}")


def generate_hierarchical_tree(model, output_dir="."):
    """Generate hierarchical module tree using graphviz."""
    from graphviz import Digraph

    dot = Digraph(comment="Model Architecture")
    dot.attr(rankdir="TB")
    dot.attr("node", shape="box", style="rounded,filled")
    dot.attr("graph", size="15,25", dpi="300", fontsize="10")

    colors = {
        "conv": "#FFE5B4",
        "transformer": "#B4D7FF",
        "attention": "#FFB4D7",
        "moe": "#D7FFB4",
        "fusion": "#FFD700",
    }

    # Root
    dot.node(
        "root", "TransNextConv", fillcolor="#87CEEB",
        fontsize="14", style="rounded,filled,bold",
    )

    # Main components
    for comp_id, comp_name in [
        ("cnn_path", "CNN Path"),
        ("transformer_path", "Transformer Path"),
        ("fusion_layer", "Fusion Layer"),
    ]:
        dot.node(comp_id, comp_name, fillcolor="#90EE90", fontsize="12")
        dot.edge("root", comp_id)

    # CNN path
    cnn_modules = [
        ("in_conv", "Stem Conv\n64 channels"),
        ("enc_1", "Encoder 1\n256 channels"),
        ("enc_2", "Encoder 2\n512 channels"),
        ("enc_3", "Encoder 3\n1024 channels"),
        ("bottleneck", "Bottleneck\n1024 channels"),
        ("dec_1", "Decoder 1\n512 channels"),
        ("dec_2", "Decoder 2\n256 channels"),
        ("dec_3", "Decoder 3\n64 channels"),
    ]

    for i, (module_id, module_name) in enumerate(cnn_modules):
        dot.node(
            f"cnn_{module_id}", module_name, fillcolor=colors["conv"],
        )
        if i == 0:
            dot.edge("cnn_path", f"cnn_{module_id}")
        else:
            dot.edge(
                f"cnn_{cnn_modules[i - 1][0]}", f"cnn_{module_id}",
            )

    # Transformer path
    transformer_modules = [
        ("crossvit", "CrossViT\nMulti-scale Encoding"),
        ("trans_enc", "Transformer Encoder\nMoE Layers"),
        ("dec_layer_1", "Decoder Layer 1\nCross Attention + MoE"),
        ("dec_layer_2", "Decoder Layer 2\nCross Attention + MoE"),
        ("dec_layer_3", "Decoder Layer 3\nCross Attention + MoE"),
    ]

    for i, (module_id, module_name) in enumerate(transformer_modules):
        color = (
            colors["transformer"]
            if "crossvit" in module_id or "trans_enc" in module_id
            else colors["attention"]
        )
        dot.node(f"trans_{module_id}", module_name, fillcolor=color)
        if i == 0:
            dot.edge("transformer_path", f"trans_{module_id}")
        else:
            dot.edge(
                f"trans_{transformer_modules[i - 1][0]}",
                f"trans_{module_id}",
            )

    # Fusion
    dot.node(
        "fusion_concat", "Concatenate\nCNN + Transformer",
        fillcolor=colors["fusion"],
    )
    dot.node(
        "fusion_conv", "Fusion Conv\n64 channels",
        fillcolor=colors["fusion"],
    )
    dot.node(
        "out_conv", "Output Conv\nn_classes",
        fillcolor=colors["fusion"],
    )

    dot.edge("fusion_layer", "fusion_concat")
    dot.edge("fusion_concat", "fusion_conv")
    dot.edge("fusion_conv", "out_conv")

    dot.edge(
        "cnn_dec_3", "fusion_concat", style="dashed", color="blue",
    )
    dot.edge(
        "trans_dec_layer_3", "fusion_concat", style="dashed", color="red",
    )

    for fmt in ["png", "svg"]:
        dot.format = fmt
        filepath = os.path.join(
            output_dir, "model_architecture_hierarchical",
        )
        dot.render(filepath, cleanup=True)
        print(f"Saved: {filepath}.{fmt}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(CONFIG)
    model.eval()

    generate_model_summary(model, device)
    generate_computational_graph(model, device)
    generate_hierarchical_tree(model)
