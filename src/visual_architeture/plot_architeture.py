import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torchinfo import summary as torchinfo_summary
import gc
from torchviz import make_dot
from scripts.train import model, output # Import output này là sai, bởi vì output không phải là global trong train.py


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Dọn dẹp bộ nhớ
gc.collect()
torch.cuda.empty_cache()

# 2. Tạo một lớp Wrapper để bọc Model lại
# Lớp này giúp kích hoạt autocast khi torchinfo chạy qua
class AutocastWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Kích hoạt Mixed Precision ngay trong forward
        with autocast():
            return self.model(x)

# 3. Chuẩn bị dữ liệu
# Lưu ý: Input tạo ra phải là .half() (FP16) để khớp với layer đầu tiên của model
dummy_input = torch.randn(1, 3, 256, 256).to(device).half()

# 4. Bọc model hiện tại vào Wrapper
# model ở đây là model đã được convert_to_fp16 (Semi-manual)
model_wrapped = AutocastWrapper(model)

# 5. Chạy torchinfo trên model đã bọc
# Lưu ý: Dùng input_data thay vì input_size để tránh Warning của torchinfo với FP16
print("\n" + "="*60)
print("FP16 (MIXED PRECISION) MODEL SUMMARY")
print("="*60)

stats_fp16 = torchinfo_summary(
    model_wrapped,
    input_data=dummy_input, # Dùng input_data trực tiếp để tránh lỗi Warning
    col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
    depth=4,
    device=device,
    verbose=1
)

# In ra tổng kết bộ nhớ params để bạn so sánh
params_size_fp16 = stats_fp16.total_param_bytes / (1024 ** 2)
print(f"\n>>> Tổng dung lượng Params thực tế (FP16 + FP32 Mixed): {params_size_fp16:.2f} MB")

params_size_fp16 = stats_fp16.total_param_bytes / (1024 ** 2)
print(f"Tổng dung lượng Params (FP16): {params_size_fp16:.2f} MB")



# Clear memory
gc.collect()
torch.cuda.empty_cache()

"""### METHOD 1: TorchViz - Computational Graph (Most Detailed)

"""

print("Generating computational graph visualization...")

# Create visualization
dot = make_dot(
    output,
    params=dict(model.named_parameters()),
    show_attrs=True,
    show_saved=False
)

# Customize appearance
dot.attr(rankdir='TB')  # Top to Bottom layout
dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
dot.attr('graph', size='20,20', dpi='300')

# Save as high-quality image
dot.format = 'png'
dot.render('model_architecture_full', cleanup=True)
print("Saved: model_architecture_full.png")

# Save as SVG (vector graphics - best quality)
dot.format = 'svg'
dot.render('model_architecture_full', cleanup=True)
print("Saved: model_architecture_full.svg (vector graphics)")

# Save as PDF
dot.format = 'pdf'
dot.render('model_architecture_full', cleanup=True)
print("Saved: model_architecture_full.pdf")

"""### METHOD 2: Simplified Version (Easier to Read)"""

print("\nGenerating simplified visualization...")

# Create a simpler version focusing on main modules
dot_simple = make_dot(
    output,
    params={name: param for name, param in model.named_parameters()
            if any(key in name for key in ['in_conv', 'enc_', 'dec_', 'bottleneck',
                                           'transformer_encoder', 'decoder_layer',
                                           'fusion', 'out_conv'])},
    show_attrs=False,
    show_saved=False
)

dot_simple.attr(rankdir='TB')
dot_simple.attr('node', shape='box', style='rounded,filled', fillcolor='lightgreen')
dot_simple.attr('graph', size='15,20', dpi='300')

dot_simple.format = 'png'
dot_simple.render('model_architecture_simplified', cleanup=True)
print("✓ Saved: model_architecture_simplified.png")

"""### METHOD 3: Using torchview (Alternative - Cleaner Layout)"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import gc

# 1. Định nghĩa Wrapper (nếu bạn chưa định nghĩa ở bước trước)
class AutocastWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        with autocast():
            return self.model(x)

try:
    # Cài đặt nếu chưa có
    # !pip install -q torchview
    from torchview import draw_graph

    print("\nGenerating torchview visualization...")

    # 2. Dọn dẹp bộ nhớ
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Chuẩn bị Input chuẩn FP16
    # Lưu ý: Input này phải nằm trên cùng device với model
    dummy_input = torch.randn(1, 3, 256, 256).to(device).half()

    # 4. Bọc Model
    # Việc này giúp torchview khi chạy qua model sẽ luôn có autocast bảo vệ
    model_wrapped = AutocastWrapper(model)

    # 5. Vẽ đồ thị
    model_graph = draw_graph(
        model_wrapped,
        input_data=dummy_input, # QUAN TRỌNG: Truyền input_data trực tiếp thay vì input_size
        device=device,
        depth=3,
        expand_nested=True,
        graph_name='TransNextConv',
        roll=True,
        save_graph=True,
        filename='model_architecture_torchview',
        directory='.'
    )

    print("Saved: model_architecture_torchview.png")
    print("Saved: model_architecture_torchview.gv") # File graphviz source

except Exception as e:
    print(f"Torchview visualization skipped: {e}")
    # In chi tiết lỗi để debug nếu vẫn hỏng
    import traceback
    traceback.print_exc()

"""### METHOD 4: Custom Hierarchical Visualization"""

print("\nGenerating hierarchical module tree...")

from graphviz import Digraph

def create_module_tree(model, name='TransNextConv'):
    """Create a hierarchical tree of model modules"""
    dot = Digraph(comment='Model Architecture')
    dot.attr(rankdir='TB')
    dot.attr('node', shape='box', style='rounded,filled')
    dot.attr('graph', size='15,25', dpi='300', fontsize='10')

    # Color scheme for different module types
    colors = {
        'conv': '#FFE5B4',  # Peach
        'transformer': '#B4D7FF',  # Light blue
        'attention': '#FFB4D7',  # Pink
        'moe': '#D7FFB4',  # Light green
        'norm': '#E5E5E5',  # Gray
        'fusion': '#FFD700',  # Gold
    }

    def get_color(name):
        name_lower = name.lower()
        if 'conv' in name_lower or 'enc_' in name_lower or 'dec_' in name_lower:
            return colors['conv']
        elif 'transformer' in name_lower or 'crossvit' in name_lower:
            return colors['transformer']
        elif 'attention' in name_lower:
            return colors['attention']
        elif 'moe' in name_lower or 'expert' in name_lower:
            return colors['moe']
        elif 'norm' in name_lower:
            return colors['norm']
        elif 'fusion' in name_lower:
            return colors['fusion']
        return '#FFFFFF'

    # Add root node
    dot.node('root', name, fillcolor='#87CEEB', fontsize='14', style='rounded,filled,bold')

    # Add main components
    main_components = [
        ('cnn_path', 'CNN Path'),
        ('transformer_path', 'Transformer Path'),
        ('fusion_layer', 'Fusion Layer')
    ]

    for comp_id, comp_name in main_components:
        dot.node(comp_id, comp_name, fillcolor='#90EE90', fontsize='12')
        dot.edge('root', comp_id)

    # CNN Path details
    cnn_modules = [
        ('in_conv', 'Stem Conv\n64 channels'),
        ('enc_1', 'Encoder 1\n256 channels'),
        ('enc_2', 'Encoder 2\n512 channels'),
        ('enc_3', 'Encoder 3\n1024 channels'),
        ('bottleneck', 'Bottleneck\n1024 channels'),
        ('dec_1', 'Decoder 1\n512 channels'),
        ('dec_2', 'Decoder 2\n256 channels'),
        ('dec_3', 'Decoder 3\n64 channels'),
    ]

    for i, (module_id, module_name) in enumerate(cnn_modules):
        dot.node(f'cnn_{module_id}', module_name, fillcolor=colors['conv'])
        if i == 0:
            dot.edge('cnn_path', f'cnn_{module_id}')
        else:
            dot.edge(f'cnn_{cnn_modules[i-1][0]}', f'cnn_{module_id}')

    # Transformer Path details
    transformer_modules = [
        ('crossvit', 'CrossViT\nMulti-scale Encoding'),
        ('trans_enc', 'Transformer Encoder\nMoE Layers'),
        ('dec_layer_1', 'Decoder Layer 1\nCross Attention + MoE'),
        ('dec_layer_2', 'Decoder Layer 2\nCross Attention + MoE'),
        ('dec_layer_3', 'Decoder Layer 3\nCross Attention + MoE'),
    ]

    for i, (module_id, module_name) in enumerate(transformer_modules):
        color = colors['transformer'] if 'crossvit' in module_id or 'trans_enc' in module_id else colors['attention']
        dot.node(f'trans_{module_id}', module_name, fillcolor=color)
        if i == 0:
            dot.edge('transformer_path', f'trans_{module_id}')
        else:
            dot.edge(f'trans_{transformer_modules[i-1][0]}', f'trans_{module_id}')

    # Fusion details
    dot.node('fusion_concat', 'Concatenate\nCNN + Transformer', fillcolor=colors['fusion'])
    dot.node('fusion_conv', 'Fusion Conv\n64 channels', fillcolor=colors['fusion'])
    dot.node('out_conv', 'Output Conv\nn_classes', fillcolor=colors['fusion'])

    dot.edge('fusion_layer', 'fusion_concat')
    dot.edge('fusion_concat', 'fusion_conv')
    dot.edge('fusion_conv', 'out_conv')

    # Connect paths to fusion
    dot.edge('cnn_dec_3', 'fusion_concat', style='dashed', color='blue')
    dot.edge('trans_dec_layer_3', 'fusion_concat', style='dashed', color='red')

    return dot

# Generate hierarchical tree
dot_tree = create_module_tree(model)
dot_tree.format = 'png'
dot_tree.render('model_architecture_hierarchical', cleanup=True)
print("Saved: model_architecture_hierarchical.png")

dot_tree.format = 'svg'
dot_tree.render('model_architecture_hierarchical', cleanup=True)
print("Saved: model_architecture_hierarchical.svg")

"""### Display file sizes"""

import os
files = [
    'model_architecture_full.png',
    'model_architecture_full.svg',
    'model_architecture_full.pdf',
    'model_architecture_simplified.png',
    'model_architecture_hierarchical.png',
    'model_architecture_hierarchical.svg'
]

for filename in files:
    if os.path.exists(filename):
        size = os.path.getsize(filename) / 1024  # KB
        print(f"{filename:45} ({size:>8.2f} KB)")

print("\nTIP: Open .svg files in browser for best quality (infinite zoom!)")
print("TIP: Use .pdf for LaTeX/papers")
