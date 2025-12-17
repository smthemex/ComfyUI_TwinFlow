import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import numpy as np
import random
import os
import argparse
import json
from torch.backends import cudnn
from methodes import TwinFlow 
from networks import DiffusionUNet

def get_args():
    parser = argparse.ArgumentParser(description="TwinFlow MNIST Training")
    
    # --- Core Parameters ---
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    
    # --- Model Parameters ---
    parser.add_argument('--conv_hidden_dim', type=int, default=64, help='UNet hidden dimension')
    parser.add_argument('--time_embed_dim', type=int, default=64, help='Time embedding dimension')
    
    # --- TwinFlow Parameters ---
    parser.add_argument('--estimate_order', type=int, default=2, help='Estimate order for RCGM loss')
    parser.add_argument('--ema_decay_rate', type=float, default=0.99, help='EMA decay rate')
    parser.add_argument('--enhanced_ratio', type=float, default=0.5, help='Training time CFG ratio')
    parser.add_argument('--using_twinflow', action='store_true', help='If using TwinFlow loss (or only RCGM loss)')
    
    # --- Miscellaneous ---
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu_id', type=str, default="0", help='GPU ID')
    parser.add_argument('--save_dir', type=str, default='./outputs/mnist', help='Directory to save logs and images')
    parser.add_argument('--data_root', type=str, default='../buffers', help='Path to MNIST dataset')
    
    return parser.parse_args()

def get_experiment_names(args):
    """
    Generate experiment names. Short name format strictly aligns with Moons example.
    """
    def fmt(val):
        return str(val).replace('.', 'p')

    short_name = f"lr={fmt(args.lr)}_edr={fmt(args.ema_decay_rate)}_eo={args.estimate_order}"
    
    # Long name includes more details for Log indexing
    long_name = (f"TwinFlow_MNIST_{short_name}_"
                 f"ep={args.epochs}_bs={args.batch_size}_"
                 f"seed={args.seed}")
    
    return long_name, short_name

def save_experiment_log(long_name, result_data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}
    
    data[long_name] = result_data
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Log updated: {filepath}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)

def main():
    args = get_args()
    set_seed(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    long_exp_name, short_exp_name = get_experiment_names(args)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device}")
    print(f"Short Name: {short_exp_name}")
    print(f"Long Name:  {long_exp_name}")

    # 1. Data Preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root=args.data_root, train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # 2. Model and Trainer
    model = DiffusionUNet(
        data_dim=784, 
        conv_hidden_dim=args.conv_hidden_dim, 
        time_embed_dim=args.time_embed_dim, 
        num_classes=10, 
        label_embed_dim=64
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    trainer = TwinFlow(
        ema_decay_rate = args.ema_decay_rate,
        estimate_order = args.estimate_order,
        enhanced_ratio = args.enhanced_ratio,
        using_twinflow = args.using_twinflow,
    )

    # 3. Training Loop
    print("Starting training...")
    final_loss = 0.0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for i, (real_img, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            real_x = real_img.view(real_img.size(0), -1).to(device)
            labels = labels.to(device)
            uncond = labels[torch.randperm(labels.size(0))]
            
            loss = trainer.training_step(model, real_x, [labels], [uncond])
            loss.backward()
            
            # Gradient handling
            for param in model.parameters():
                if param.grad is not None:
                    torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        final_loss = epoch_loss/len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {final_loss:.4f}")

    # 4. Save Log
    result_data = {
        "final_loss": final_loss,
        "args": vars(args)
    }
    save_experiment_log(long_exp_name, result_data, os.path.join(args.save_dir, "experiments_log.json"))

    # 5. Visualization (10x10 Ordered Grid)
    # Uses save_image directly to avoid matplotlib margins and titles
    print("Generating 10x10 Ordered Visualization...")
    n_classes = 10
    n_samples_per_class = 10
    total_vis = n_classes * n_samples_per_class
    
    with torch.no_grad():
        # Construct ordered labels: 10 of 0, 10 of 1...
        vis_labels = torch.arange(n_classes).repeat_interleave(n_samples_per_class).to(device)
        
        z_vis = torch.randn(total_vis, 784).to(device)
        
        # Condition Generation using UCGM sampler
        gen_vis = trainer.sampling_loop(z_vis, model, **dict(c=[vis_labels]))[-1]
        
        # Restore to image space
        # Map from [-1, 1] back to [0, 1]
        gen_vis = (gen_vis + 1) / 2.0
        gen_vis = gen_vis.clamp(0, 1).view(total_vis, 1, 28, 28)
        
        save_path = os.path.join(args.save_dir, f"{short_exp_name}.png")
        
        # Save image directly using torchvision utils
        # nrow=10 ensures 10 digits per row
        # padding=2 sets the interval between digits
        utils.save_image(gen_vis, save_path, nrow=n_samples_per_class, padding=2)
        
        print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    main()
