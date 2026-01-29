import torch
import sys
import os
import argparse
from pathlib import Path

# 1. Import your actual model class
try:
    from model import BehaviorSequenceClassifier
except ImportError:
    print("Error: Could not import 'model.py'. Make sure this script is in the same folder as your source code.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Quantize a trained BehaviorScope LSTM model for CPU inference.")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the trained .pt checkpoint.")
    parser.add_argument("--output_path", type=Path, default=None, help="Path for the output quantized model.")
    return parser.parse_args()

def main():
    args = parse_args()
    model_path = args.model_path

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    print(f"Loading checkpoint from: {model_path}")

    # 2. Load the Checkpoint
    # We use weights_only=False because we need to support legacy pickles if present, 
    # but primarily to load the dictionary structure.
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    # Check if it's a dict (standard checkpoint) or already a model object
    if not isinstance(checkpoint, dict):
        print("Error: The provided file seems to be a model object, not a checkpoint dictionary. It might already be quantized.")
        sys.exit(1)

    # 3. Extract Metadata
    hparams = checkpoint.get("hparams", {})
    idx_to_class = checkpoint.get("idx_to_class") or hparams.get("idx_to_class")

    if idx_to_class is None:
        print("Error: Could not find class names (idx_to_class) in checkpoint.")
        sys.exit(1)

    # Ensure keys are integers
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    num_classes = len(idx_to_class)
    print(f"Configuration: {hparams.get('backbone')} + {hparams.get('sequence_model')}")
    print(f"Classes found ({num_classes}): {list(idx_to_class.values())}")

    # 4. Initialize the Architecture
    model = BehaviorSequenceClassifier(
        num_classes=num_classes,
        backbone=hparams.get("backbone", "mobilenet_v3_small"),
        pretrained_backbone=False,
        train_backbone=False,
        hidden_dim=int(hparams.get("hidden_dim", 256)),
        num_layers=int(hparams.get("num_layers", 1)),
        dropout=float(hparams.get("dropout", 0.2)),
        bidirectional=bool(hparams.get("bidirectional", False)),
        sequence_model=hparams.get("sequence_model", "lstm"),
        temporal_attention_layers=int(hparams.get("temporal_attention_layers", 0)),
        attention_heads=int(hparams.get("attention_heads", 4)),
        positional_encoding=hparams.get("positional_encoding", "none"),
        positional_encoding_max_len=int(hparams.get("positional_encoding_max_len", 512)),
        use_attention_pooling=bool(hparams.get("attention_pool", False)),
        use_feature_se=bool(hparams.get("feature_se", False)),
        slowfast_alpha=int(hparams.get("slowfast_alpha", 4)),
        slowfast_fusion_ratio=float(hparams.get("slowfast_fusion_ratio", 0.25)),
        slowfast_base_channels=int(hparams.get("slowfast_base_channels", 48)),
    )

    # 5. Load Weights
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # 6. [CRITICAL STEP] Attach Metadata to the Object
    #    This "tattoos" the class info onto the object so it survives torch.save()
    model.idx_to_class = idx_to_class
    model.num_classes = num_classes
    # We also attach hparams just in case future scripts need them
    model.hparams = hparams 

    print("Metadata attached to model object.")

    # 7. Apply Dynamic Quantization
    print("Quantizing model (Dynamic - LSTM/Linear)...")
    # Note: Dynamic quantization primarily benefits CPU inference.
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.LSTM, torch.nn.Linear}, 
        dtype=torch.qint8
    )

    # 8. Save
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = model_path.with_name(f"{model_path.stem}_quantized.pt")

    torch.save(quantized_model, output_path)

    print(f"Success! Saved to: {output_path}")
    print(f"Original size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    print(f"Quantized size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()