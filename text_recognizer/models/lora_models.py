import torch
import torch.nn as nn
import torch.optim as optim
from peft import LoraConfig, get_peft_model
from text_recognizer.models.resnet_transformer import ResnetTransformer
from lightning_transformer import TransformerLitModel


def build_lora_model(data_config, checkpoint_path: str, device: str = "cuda") -> nn.Module:
    # Transformer settings
    args = _default_args()

    # Instantiate base model
    model = ResnetTransformer(data_config, args).to(device)

    # Load checkpoint and update model state dict
    model_script = torch.jit.load(checkpoint_path, map_location=device)
    state_dict = model_script.state_dict()
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "self_attn.in_proj_weight",
            "self_attn.out_proj",
            "multihead_attn.in_proj_weight",
            "multihead_attn.out_proj",
            "linear1",
            "linear2",
        ],
    )

    # Wrap with PEFT
    lora_model = get_peft_model(model, lora_config)
    lit_model = TransformerLitModel(lora_model, args)

    return lit_model


def _default_args():
    import argparse

    TF_DIM = 256
    TF_FC_DIM = 256
    TF_DROPOUT = 0.4
    TF_LAYERS = 4
    TF_NHEAD = 4

    return argparse.Namespace(
        tf_dim=TF_DIM,
        tf_fc_dim=TF_FC_DIM,
        tf_nhead=TF_NHEAD,
        tf_dropout=TF_DROPOUT,
        tf_layers=TF_LAYERS,
    )


def get_loss_and_optimizer(model, data_config, lr=1e-4):
    criterion = nn.CrossEntropyLoss(ignore_index=data_config["inverse_mapping"]["<P>"])
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    return criterion, optimizer
