
from argparse import ArgumentParser
import sys
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from smart.utils.config import load_config_act
from smart.datamodules import MultiDataModule
from smart.model import SMART
from smart.utils.log import Logging


if __name__ == '__main__':
    parser = ArgumentParser()
    Predictor_hash = {"smart": SMART, }
    parser.add_argument('--config', type=str, default='configs/train/train_overfitting_workflow.yaml')
    parser.add_argument('--pretrain_ckpt', type=str, default="")
    parser.add_argument('--ckpt_path', type=str, default="")
    parser.add_argument('--save_ckpt_path', type=str, default="")
    parser.add_argument('--min_acc', type=float, default=0.2,
                        help='Minimum required validation accuracy to pass (float)')
    # Add --cpu flag
    parser.add_argument('--cpu', action='store_true', help='Force training on CPU')
    args = parser.parse_args()
    config = load_config_act(args.config)
    Predictor = Predictor_hash[config.Model.predictor]
    strategy = DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
    Data_config = config.Dataset
    datamodule = MultiDataModule(**vars(Data_config))

    if args.pretrain_ckpt == "":
        model = Predictor(config.Model)
    else:
        logger = Logging().log(level='DEBUG')
        model = Predictor(config.Model)
        model.load_params_from_file(filename=args.pretrain_ckpt,
                                    logger=logger)
    trainer_config = config.Trainer
    
    if args.cpu:
        trainer_config.accelerator = "cpu"
    
    model_checkpoint = ModelCheckpoint(dirpath=args.save_ckpt_path,
                                       filename="{epoch:02d}",
                                       monitor='val_cls_acc',
                                       every_n_epochs=1,
                                       save_top_k=5,
                                       mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator=trainer_config.accelerator, devices=trainer_config.devices,
                         strategy=strategy,
                         accumulate_grad_batches=trainer_config.accumulate_grad_batches,
                         num_nodes=trainer_config.num_nodes,
                         callbacks=[model_checkpoint, lr_monitor],
                         max_epochs=trainer_config.max_epochs,
                         num_sanity_val_steps=0,
                         gradient_clip_val=0.5)
    if args.ckpt_path == "":
        trainer.fit(model,
                    datamodule)
    else:
        trainer.fit(model,
                    datamodule,
                    ckpt_path=args.ckpt_path)

# After training, check best accuracy against threshold and fail if below
best_score = None
if hasattr(model_checkpoint, 'best_model_score') and model_checkpoint.best_model_score is not None:
    val = model_checkpoint.best_model_score
    try:
        best_score = float(val.item()) if hasattr(val, 'item') else float(val)
    except Exception:
        try:
            best_score = float(val)
        except Exception:
            best_score = None
else:
    metrics = getattr(trainer, 'callback_metrics', {})
    if metrics and 'val_cls_acc' in metrics and metrics['val_cls_acc'] is not None:
        val = metrics['val_cls_acc']
        try:
            best_score = float(val.item()) if hasattr(val, 'item') else float(val)
        except Exception:
            try:
                best_score = float(val)
            except Exception:
                best_score = None

if best_score is None:
    print('Warning: Could not determine final accuracy; skipping threshold check')
else:
    print(f"Best accuracy: {best_score:.6f} (threshold: {args.min_acc})")
    if best_score < args.min_acc:
        print('Accuracy below threshold, failing job')
        sys.exit(1)
    else:
        print('Accuracy meets threshold')
