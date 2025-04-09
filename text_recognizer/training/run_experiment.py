import argparse
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only
import torch
from text_recognizer import callbacks as cb

from text_recognizer.data.create_save_argument_dataset import (DL_DATA_DIRNAME,
                                                               inverse_mapping,
                                                               mapping)

import text_recognizer.metadata.iam_paragraphs as metadata_iam_paragraphs

from text_recognizer.models.lora_models import build_lora_model
from text_recognizer.data.data_module import ArgumentDataModule
from lightning.pytorch.profilers import PyTorchProfiler, PassThroughProfiler
from torch.profiler import ProfilerActivity


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments manually
    parser.add_argument('--max_epochs', type=int, default=1, help='Max number of epochs')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--precision', type=int, default=16, help='Precision of the model')
    parser.add_argument('--limit_train_batches', type=float, default=0.1, help='Limit number of training batches')
    parser.add_argument('--limit_val_batches', type=float, default=0.1, help='Limit number of validation batches')
    parser.add_argument('--log_every_n_steps', type=int, default=10, help='Logging frequency in steps')
    parser.add_argument('--wandb', action='store_true', help="Use Weights & Biases for logging")
    parser.add_argument('--devices', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--data_dir', type=str, default=str(DL_DATA_DIRNAME), help='Number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=32, help='number of data points in a batch')

    parser.add_argument('--tf_dim', type=int, default=256, help='embedding dimension for transformer')
    parser.add_argument('--tf_fc_dim', type=int, default=256, help='classifier dimension for transformer')
    parser.add_argument('--tf_nhead', type=int, default=4, help='number of attention heads')
    parser.add_argument('--tf_dropout', type=int, default=0.4, help='Dropout rate')
    parser.add_argument('--tf_layers', type=int, default=4, help='number of transformer layers')
    parser.add_argument('--input_dims', type=int, default=metadata_iam_paragraphs.DIMS, help='input dimensions')
    parser.add_argument('--output_dims', type=int, default=metadata_iam_paragraphs.OUTPUT_DIMS, help='output tokens dimensions')
    parser.add_argument('--checkpoint_path', type=str, default=r"D:\RL_Finance\MLops\fslab\lab07\text_recognizer\artifacts\paragraph-text-recognizer\model.pt", help='pre-trained model checkpoint')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dataset_len', type=int, default=1000, help='learning rate')



    # Basic arguments
    parser.add_argument(
        "--check_val_every_n_epoch", 
        type=int, 
        default=1, 
        help="Number of epochs between validation checks"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="If passed, uses the PyTorch Profiler to track computation, exported as a Chrome-style trace.",
    )


    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (cuda or cpu). Defaults to cuda if available."
    )
    
    parser.add_argument(
        "--stop_early",
        type=int,
        default=0,
        help="If non-zero, applies early stopping, with the provided value as the 'patience' argument."
        + " Default is 0.",
    )

    parser.add_argument("--help", "-h", action="help")
    return parser


@rank_zero_only
def _ensure_logging_dir(experiment_dir):
    """Create the logging directory via the rank-zero process, if necessary."""
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)


def main():
    """
    Run an experiment.

    Sample command:
    ```
    python training/run_experiment.py --max_epochs=3 --gpus='0,' --num_workers=20 --model_class=TransformerLitModel --data_class=CustomDataModule
    ```

    For basic help documentation, run the command
    ```
    python training/run_experiment.py --help
    ```

    """
    parser = _setup_parser()
    args = parser.parse_args()



    # Setup the data and model from parsed args
    data = ArgumentDataModule(data_dir=args.data_dir, dataset_len=args.dataset_len, batch_size=args.batch_size, val_split=0.2, num_workers=4)
    
    # Define data config (typically would come from your dataset's configuration)

    data_config = {
    "input_dims": args.input_dims,  # (channels, height, width)
    "output_dims": args.output_dims,  # Maximum output sequence length
    "mapping": mapping,  # Example mapping for digits
    "inverse_mapping": inverse_mapping,
    }

    lora_model = build_lora_model(data_config, args.checkpoint_path, args.device)

    # Setup logging directory
    log_dir = Path("training") / "logs"
    _ensure_logging_dir(log_dir)
    logger = pl.loggers.TensorBoardLogger(log_dir)
    experiment_dir = logger.log_dir

    # Metric for model monitoring
    goldstar_metric = "validation/cer"
    filename_format = "epoch={epoch:04d}-validation.loss={validation/loss:.3f}"
    if goldstar_metric == "validation/cer":
        filename_format += "-validation.cer={validation/cer:.3f}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=5,
        filename=filename_format,
        monitor=goldstar_metric,
        mode="min",
        auto_insert_metric_name=False,
        dirpath=experiment_dir,
        every_n_epochs=args.check_val_every_n_epoch,
    )

    summary_callback = pl.callbacks.ModelSummary(max_depth=2)

    callbacks = [summary_callback, checkpoint_callback]

    # Setup WandB if enabled
    if args.wandb:
        logger = pl.loggers.WandbLogger(log_model="all", save_dir=str(log_dir), job_type="train")
        logger.watch(lora_model, log_freq=max(100, args.log_every_n_steps))
        logger.log_hyperparams(vars(args))
        experiment_dir = logger.experiment.dir

    callbacks += [cb.ModelSizeLogger(), cb.LearningRateMonitor()]


    if args.wandb:
        callbacks.append(cb.ImageToTextLogger())


    
    if args.profile:
        profiler = PyTorchProfiler(export_to_chrome=True,
                                   activities=[
                                        ProfilerActivity.CPU, 
                                        ProfilerActivity.CUDA
                                    ],
                                    use_cuda=True,
                                    dirpath=experiment_dir,
                                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
                                    )
    else:
        profiler = PassThroughProfiler()  # Don't use profiler if not enabled
 


    trainer = pl.Trainer(
                max_epochs=args.max_epochs,
                accelerator="gpu",
                devices=args.devices,
                precision=args.precision,
                limit_train_batches=args.limit_train_batches,
                limit_val_batches=args.limit_val_batches,
                logger=logger,
                check_val_every_n_epoch=args.check_val_every_n_epoch,
                enable_progress_bar=False,
                callbacks=callbacks,
                profiler=profiler
                )

    trainer.profiler = profiler

    # Train the model
    trainer.fit(lora_model, datamodule=data)


if __name__ == "__main__":
    main()
