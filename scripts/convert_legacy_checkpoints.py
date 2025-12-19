"""
Convert legacy single-file checkpoints (.pt) to new JSON-based format.

Usage:
    python convert_legacy_checkpoints.py --model_name dante
    python convert_legacy_checkpoints.py --model_name dante --checkpoint final.pt
    python convert_legacy_checkpoints.py --all
"""

import os
import argparse
import torch
from core.checkpoint import CheckpointManager
from core.model import GPT


def convert_checkpoint(model_name, legacy_file="latest.pt", dry_run=False):
    """
    Convert a legacy checkpoint to new JSON format.
    
    Args:
        model_name: Name of the model (e.g., "dante")
        legacy_file: Legacy checkpoint filename (e.g., "latest.pt", "final.pt")
        dry_run: If True, only show what would be converted without actually converting
    
    Returns:
        True if conversion successful, False otherwise
    """
    ckpt_manager = CheckpointManager(model_name)
    legacy_path = ckpt_manager.get_path(legacy_file)
    
    # Check if legacy checkpoint exists
    if not os.path.exists(legacy_path):
        print(f"❌ No legacy checkpoint found: {legacy_path}")
        return False
    
    # Check if already converted
    if ckpt_manager.exists_new_format():
        print(f"ℹ️  Model '{model_name}' already has new format checkpoint")
        print(f"   Legacy file: {legacy_path}")
        print(f"   New format at: {ckpt_manager.checkpoint_dir}")
        response = input("   Convert anyway? This will overwrite new format files. (y/N): ")
        if response.lower() != 'y':
            return False
    
    if dry_run:
        print(f"[DRY RUN] Would convert: {legacy_path}")
        return True
    
    print(f"Converting {model_name}/{legacy_file}...")
    
    try:
        # Load legacy format
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, _, step, optim_state = GPT.load_legacy(legacy_path, map_location=device)
        
        # Display info
        print(f"  Config: {model.config}")
        print(f"  Step: {step}")
        print(f"  Has optimizer: {optim_state is not None}")
        
        # Save in new format
        model.save_checkpoint_bundle(
            ckpt_manager.checkpoint_dir,
            step=step,
            optimizer_state=optim_state
        )
        
        print(f"✅ Successfully converted to new format!")
        print(f"   Location: {ckpt_manager.checkpoint_dir}")
        print(f"   Files created:")
        print(f"     - model.pt")
        print(f"     - config.json")
        print(f"     - tokenizer.json")
        if step is not None or optim_state is not None:
            print(f"     - training_state.json")
        if optim_state is not None:
            print(f"     - optimizer.pt")
        
        # Ask about deleting legacy file
        print(f"\n   Legacy file: {legacy_path}")
        response = input(f"   Keep legacy file as backup? (Y/n): ")
        if response.lower() == 'n':
            os.remove(legacy_path)
            print(f"   Deleted legacy file: {legacy_path}")
        else:
            print(f"   Kept legacy file as backup")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_all_legacy_checkpoints(base_dir="checkpoints"):
    """
    Find all legacy checkpoints in the checkpoints directory.
    
    Returns:
        List of tuples: (model_name, checkpoint_filename)
    """
    legacy_checkpoints = []
    
    if not os.path.exists(base_dir):
        return legacy_checkpoints
    
    for model_name in os.listdir(base_dir):
        model_dir = os.path.join(base_dir, model_name)
        
        if not os.path.isdir(model_dir):
            continue
        
        # Check for .pt files
        for filename in os.listdir(model_dir):
            if filename.endswith(".pt"):
                # Check if it's a legacy checkpoint (has "config" key)
                try:
                    path = os.path.join(model_dir, filename)
                    # Legacy checkpoints contain dicts with config/tokenizer
                    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                    if "config" in checkpoint and "model_state_dict" in checkpoint:
                        legacy_checkpoints.append((model_name, filename))
                except:
                    pass
    
    return legacy_checkpoints


def main():
    from core.banner import print_banner
    print_banner("MyPT Converter", "Legacy Checkpoint Migration")
    
    parser = argparse.ArgumentParser(
        description="Convert legacy checkpoints to new JSON-based format"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model to convert (e.g., dante, shakespeare)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest.pt",
        help="Legacy checkpoint filename (default: latest.pt)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all legacy checkpoints found"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be converted without actually converting"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Legacy Checkpoint Converter")
    print("=" * 60)
    print()
    
    if args.all:
        # Find and convert all legacy checkpoints
        legacy_checkpoints = find_all_legacy_checkpoints()
        
        if not legacy_checkpoints:
            print("✅ No legacy checkpoints found. All models are using new format!")
            return
        
        print(f"Found {len(legacy_checkpoints)} legacy checkpoint(s):")
        for model_name, filename in legacy_checkpoints:
            print(f"  - {model_name}/{filename}")
        print()
        
        if not args.dry_run:
            response = input(f"Convert all {len(legacy_checkpoints)} checkpoint(s)? (y/N): ")
            if response.lower() != 'y':
                print("Cancelled.")
                return
        
        print()
        successful = 0
        for model_name, filename in legacy_checkpoints:
            if convert_checkpoint(model_name, filename, args.dry_run):
                successful += 1
            print()
        
        print("=" * 60)
        print(f"Conversion complete: {successful}/{len(legacy_checkpoints)} successful")
        print("=" * 60)
    
    elif args.model_name:
        # Convert specific model
        success = convert_checkpoint(args.model_name, args.checkpoint, args.dry_run)
        if success:
            print()
            print("=" * 60)
            print("✅ Conversion complete!")
            print("=" * 60)
        else:
            print()
            print("=" * 60)
            print("❌ Conversion failed or cancelled")
            print("=" * 60)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

