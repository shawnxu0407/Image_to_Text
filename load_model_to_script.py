"""Stages a model for use in production.

If based on a checkpoint, the model is converted to torchscript, saved locally,
and uploaded to W&B.

If based on a model that is already converted and uploaded, the model file is downloaded locally.

For details on how the W&B artifacts backing the checkpoints and models are handled,
see the documenation for stage_model.find_artifact.
"""
import argparse
from pathlib import Path
import tempfile

import torch
import wandb

from training.util import setup_data_and_model_from_args


# these names are all set by the pl.loggers.WandbLogger
MODEL_CHECKPOINT_TYPE = "model"
BEST_CHECKPOINT_ALIAS = "best"
MODEL_CHECKPOINT_PATH = "model.ckpt"
LOG_DIR = Path("training") / "logs"

STAGED_MODEL_TYPE = "prod-ready"  # we can choose the name of this type, and ideally it's different from checkpoints
STAGED_MODEL_FILENAME = "model.pt"  # standard nomenclature; pytorch_model.bin is also used

PROJECT_ROOT = Path(__file__).resolve().parents[0]

from text_recognizer.data.create_save_argument_dataset import (inverse_mapping,
                                                               mapping)
from text_recognizer.models.resnet_transformer import ResnetTransformer
from peft import LoraConfig, get_peft_model


api = wandb.Api()

DEFAULT_ENTITY = api.default_entity
DEFAULT_FROM_PROJECT = "image_to_text"
DEFAULT_TO_PROJECT = "image_to_text"
DEFAULT_STAGED_MODEL_NAME = "text-recognizer"

PROD_STAGING_ROOT = PROJECT_ROOT / "text_recognizer" / "artifacts"


CKPT_AT = "xiangyexu-university-of-waterloo/image_to_text/model-694o33pb:best"



def main(args):
    prod_staging_directory = PROD_STAGING_ROOT
    prod_staging_directory.mkdir(exist_ok=True, parents=True)
    checkpoint_path = Path(prod_staging_directory) / "model.ckpt"
    torch_script_path = Path(prod_staging_directory) / 'model.pt'

    
    config=download_artifact(args.ckpt_at, prod_staging_directory, args.download)
    # reload the model from that checkpoint
    lora_model = load_model_from_checkpoint(config, checkpoint_path)
    # save the model to torchscript in the staging directory
    save_model_to_torchscript(lora_model, directory=torch_script_path)






def _get_entity_from(args):
    entity = args.entity
    if entity is None:
        raise RuntimeError(f"No entity argument provided. Use --entity=DEFAULT to use {DEFAULT_ENTITY}.")
    elif entity == "DEFAULT":
        entity = DEFAULT_ENTITY

    return entity







def download_artifact(artifact_path, download_dir, download=False):
    artifact = api.artifact(artifact_path)
    if download:
       artifact.download(root=download_dir)
    run = artifact.logged_by()
    config = dict(run.config)
    return config



def save_model_to_torchscript(model, directory):
    merged_model = model.merge_and_unload()
    scripted_model = torch.jit.script(merged_model)
    torch.jit.save(scripted_model, directory)
    print(f"Saved TorchScript model to {directory}")


def load_model_from_checkpoint(config: dict, checkpoint_path=None):

    model_args = argparse.Namespace(
    tf_dim=config['tf_dim'],
    tf_fc_dim=config['tf_fc_dim'],
    tf_nhead=config['tf_nhead'],
    tf_dropout=config['tf_dropout'],
    tf_layers=config['tf_layers']
    )

    # If metadata contains these, prefer them over hardcoding
    data_config = {
    "input_dims": config['input_dims'],  # (channels, height, width)
    "output_dims": config['output_dims'],  # Maximum output sequence length
    "mapping": mapping,  # Example mapping for digits
    "inverse_mapping": inverse_mapping,
    }

    target_module_1 = [
    "self_attn.in_proj_weight",
    "self_attn.out_proj",
    "multihead_attn.in_proj_weight",
    "multihead_attn.out_proj",
    "linear1",
    "linear2",
    ]

    lora_config = LoraConfig(
        r=8,   # Rank of decomposition
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.1,
        target_modules=target_module_1
    )

    model = ResnetTransformer(data_config, model_args)

    lora_model = get_peft_model(model, lora_config)
    lora_state_dict = lora_model.state_dict()

    ckpt_state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]

    ## make a new state dict with correct key names
    new_state_dict = {}
    for key1, key2 in zip(lora_state_dict.keys(), ckpt_state_dict.values()):
        new_state_dict[key1] = key2



    lora_model.load_state_dict(new_state_dict)

    return lora_model



def _setup_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help=f"If provided, check ENTITY/FROM_PROJECT for an artifact with the provided STAGED_MODEL_NAME and download its latest version to {PROD_STAGING_ROOT}/STAGED_MODEL_NAME.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help=f"Entity from which to download the checkpoint. Note that checkpoints are always uploaded to the logged-in wandb entity. Pass the value 'DEFAULT' to also download from default entity, which is currently {DEFAULT_ENTITY}.",
    )
    parser.add_argument(
        "--from_project",
        type=str,
        default=DEFAULT_FROM_PROJECT,
        help=f"Project from which to download the checkpoint. Default is {DEFAULT_FROM_PROJECT}",
    )
    parser.add_argument(
        "--to_project",
        type=str,
        default=DEFAULT_TO_PROJECT,
        help=f"Project to which to upload the compiled model. Default is {DEFAULT_TO_PROJECT}.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help=f"Optionally, the name of a run to check for an artifact of type {MODEL_CHECKPOINT_TYPE} that has the provided CKPT_ALIAS. Default is None.",
    )
    parser.add_argument(
        "--ckpt_alias",
        type=str,
        default=BEST_CHECKPOINT_ALIAS,
        help=f"Alias that identifies which model checkpoint should be staged.The artifact's alias can be set manually or programmatically elsewhere. Default is {BEST_CHECKPOINT_ALIAS!r}.",
    )
    parser.add_argument(
        "--staged_model_name",
        type=str,
        default=DEFAULT_STAGED_MODEL_NAME,
        help=f"Name to give the staged model artifact. Default is {DEFAULT_STAGED_MODEL_NAME!r}.",
    )
    parser.add_argument(
        "--download",
        type=str,
        default=False,
        help=f"whether decide to download the artifact. Default is {False}.",
    )
    parser.add_argument(
        "--ckpt_at",
        type=str,
        default=CKPT_AT,
        help=f"Location of the checkpoint. Default is {CKPT_AT!r}.",
    )


    return parser



if __name__ == "__main__":
    parser = _setup_parser()
    args = parser.parse_args()
    main(args)
