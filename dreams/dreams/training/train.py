import os
import pandas as pd
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pathlib import Path
import warnings
from numba import NumbaDeprecationWarning

warnings.filterwarnings(
    "ignore", message=".*cpp.*"
)  # Suppress internal torch TransfromerEncoder warnings
warnings.filterwarnings(
    "ignore", category=NumbaDeprecationWarning
)  # Supress numba warnings from UMAP
import dreams.utils.data as du
from dreams.utils.io import setup_logger
from dreams.models.dreams.dreams import DreaMS

# imports for Neptune
import neptune
from lightning.pytorch.loggers.neptune import NeptuneLogger
from dotenv import load_dotenv
import logging
from typing import List

# from dreams.models.vanilla_bert.bert import VanillaBERT
from dreams.models.heads.heads import *
from dreams.models.baselines.deep_sets import *
from dreams.training.train_argparse import parse_args
from dreams.utils.data import ContrastiveSpectraDataset
import torch

torch.set_printoptions(profile="full")
torch.set_float32_matmul_precision("high")
torch.cuda.empty_cache()

def init_neptune(
        tags: List[str],
        mode: str = "async",
        name: Optional[str] = None,
        api_token_var: str = "NEPTUNE_API_TOKEN",
        project_name_var: str = "NEPTUNE_PROJECT"
) -> neptune.Run:
    """
    Initialize a Neptune run using environment variables.

    Parameters:
    - tags (List[str]): Tags to associate with the Neptune run.
    - mode (str): Neptune connection mode. Defaults to "async".
                  Valid values: "async", "sync", "offline", "read-only", "debug".
    - name (Optional[str]): Optional name for the run.
    - api_token_var (str): Environment variable for Neptune API token.
    - project_name_var (str): Environment variable for Neptune project name.

    Returns:
    - neptune.Run: Initialized Neptune run object.
    """
    logger = logging.getLogger(__name__)

    logger.info("Loading environment variables...")
    load_dotenv()

    project_name = os.getenv(project_name_var)
    if not project_name:
        raise ValueError(f"Environment variable '{project_name_var}' is not set.")

    # Only require API token if not running offline
    api_token = os.getenv(api_token_var) if mode != "offline" else None
    if mode != "offline" and not api_token:
        raise ValueError(f"Environment variable '{api_token_var}' is not set for mode '{mode}'.")

    logger.info(f"Initializing Neptune run in '{mode}' mode for project '{project_name}'.")

    run = neptune.init_run(
        project=project_name,
        api_token=api_token,
        mode=mode,
        name=name,
        tags=tags,
    )

    logger.info("Neptune run initialized successfully.")
    return run

def fully_sanitize(obj):
    # Recursively convert all values to builtin Python types.
    if isinstance(obj, dict):
        return {k: fully_sanitize(v) for k, v in obj.items()}
    elif hasattr(obj, '__dict__'):
        return fully_sanitize(vars(obj))
    elif isinstance(obj, (list, tuple)):
        return [fully_sanitize(v) for v in obj]
    elif isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    else:
        return str(obj)

def main(args):
    # Prepare seeds and auxiliary variables
    seed_everything(args.seed)
    run_dir = Path(args.project_name) / args.job_key
    run_dir.mkdir(parents=True, exist_ok=True)
    args.gains_dir = run_dir / "model_gains"
    args.gains_dir.mkdir(exist_ok=True)
    logger = setup_logger(run_dir.with_suffix(".log"))
    logger.info(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the way to preprocess spectra (same for train or validation on independent datasets)
    spec_preproc = du.SpectrumPreprocessor(
        dformat=args.dformat,
        prec_intens=args.prec_intens,
        n_highest_peaks=args.max_peaks_n,
        spec_entropy_cleaning=args.spec_entropy_cleaning,
        precision=args.train_precision,
        mz_shift_aug_p=args.mz_shift_aug_p,
        mz_shift_aug_max=args.mz_shift_aug_max,
    )

    # Define datasets
    if args.train_regime == "pre-training":
        dataset = du.MaskedSpectraDataset(
            in_pth=args.dataset_pth,
            spec_preproc=spec_preproc,
            n_samples=args.n_samples,
            dformat=args.dformat,
            logger=logger,
            ssl_objective=args.train_objective,
            deterministic_mask=args.deterministic_mask,
            frac_masks=args.frac_masks,
            mask_prec=args.mask_prec,
            mask_peaks=args.mask_peaks,
            mask_intens_strategy=args.mask_intens_strategy,
            ret_order_pairs=args.ret_order_loss_w != 0,
            acc_est_weight=args.acc_est_weight,
            lsh_weight=args.lsh_weight,
            mask_val=args.mask_val,
            bert801010_masking=args.bert801010_masking,
        )

        # with h5py.File(args.dataset_pth, 'r') as f:
        #     split_col = f['val'][:] if 'val' in f.keys() else None
        split_col = None

        if split_col is not None:
            # NOTE: max_var_features is not implemented for SplittedDataModule
            data_module = du.SplittedDataModule(
                dataset,
                split_mask=split_col,
                batch_size=args.batch_size,
                num_workers=args.num_workers_data,
                seed=args.seed,
            )
        else:
            data_module = du.RandomSplitDataModule(
                dataset=dataset,
                batch_size=args.batch_size,
                val_frac=args.val_frac,
                num_workers=args.num_workers_data,
                max_var_features=dataset.data[args.max_batch_var_features]
                if args.max_batch_var_features
                else None,
            )
    elif args.train_regime in {"fine-tuning", "cv-fine-tuning"}:
        if args.dataset_pth.suffix == ".hdf5":
            msdata = du.MSData(args.dataset_pth, in_mem=True)
            dataset = msdata.to_torch_dataset(
                spec_preproc=spec_preproc,
                label=args.train_objective,
                dformat=args.dformat,
            )
            # data_module = du.SplittedDataModule(
            #     dataset=dataset,
            #     split_mask=pd.Series(msdata.get_values(FOLD)),
            #     batch_size=args.batch_size,
            #     num_workers=args.num_workers_data,
            #     n_train_samples=args.n_samples,
            #     seed=args.seed,
            # )

            # Set num_workers to 0 for HDF5 files to avoid pickling issues
            data_module = du.RandomSplitDataModule(  # RandomSplitDataModule is simpler and sufficient
                dataset=dataset,
                val_frac=args.val_frac,
                batch_size=args.batch_size,
                num_workers=0,
            )
        else:
            raise ValueError(f"Unknown dataset type: {args.dataset_pth.suffix}.")

    # Log dataset sizes
    cv = True if isinstance(data_module, du.CVDataModule) else False

    # If cross validation, iterate over folds
    for i in range(data_module.get_num_folds() if cv else 1):
        if cv:
            data_module.setup_fold_index(i)

        # Define model
        if args.model == "DreaMS":
            if not args.pre_trained_pth:
                model = DreaMS(args, spec_preproc)
        elif args.model == "DeepSets":
            if args.train_objective.startswith("fp"):
                model = DeepSetsPeaksFingerprint(args.train_objective, lr=args.lr)
            elif args.train_objective in {"num_C", "num_O"}:
                model = DeepSetsPeakIntReg(lr=args.lr)
            elif args.train_objective in {"qed", "ms2prop_labels"}:
                model = DeepSetsPeakReg(
                    lr=args.lr,
                    out_dim=10 if args.train_objective == "ms2prop_labels" else 1,
                )
            elif args.train_objective in {"has_N", "has_Cl", "has_F"}:
                model = DeepSetsPeakBinCls(lr=args.lr)
        else:
            NotImplementedError(f"Model {args.model} is not implemented")

        # Append fine-tuning heads
        if "fine-tuning" in args.train_regime and args.model != "DeepSets":
            backbone = (
                args.pre_trained_pth
                if args.pre_trained_pth is not None and len(str(args.pre_trained_pth)) > 4
                else DreaMS(args, spec_preproc)
            )
            if args.train_objective in {
                "fp_morgan_2048",
                "fp_morgan_4096",
                "fp_rdkit_2048",
                "fp_rdkit_4096",
            }:
                model = FingerprintHead(
                    backbone=backbone,
                    fp_str=args.train_objective,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    dropout=args.dropout,
                    retrieval_val_pth=args.retrieval_val_pth,
                    batch_size=args.batch_size,
                    unfreeze_backbone_at_epoch=args.unfreeze_backbone_at_epoch,
                    store_val_out_dir=run_dir / f"val_out_{args.dataset_pth.stem}",
                    head_depth=args.head_depth,
                    head_phi_depth=args.head_phi_depth,
                )
            # TODO: refactor backbone
            if args.train_objective in {"num_C", "num_O"}:
                model = IntRegressionHead(
                    args.pre_trained_pth, args.lr, args.weight_decay
                )
            elif args.train_objective in {"qed"}:
                model = RegressionHead(
                    args.pre_trained_pth, args.lr, args.weight_decay, sigmoid=True
                )
            elif args.train_objective in {"has_N", "has_Cl", "has_F"}:
                model = BinClassificationHead(
                    args.pre_trained_pth,
                    args.lr,
                    args.weight_decay,
                    focal_loss_alpha=args.focal_loss_alpha,
                    focal_loss_gamma=args.focal_loss_gamma,
                )
            elif args.train_objective == "contrastive_spec_embs":
                # df_smiles_similarities = pd.read_pickle(MERGED_DATASETS / 'nist20_clean_MoNA_contrastive_v2_10ppm_smiles_similarities_asymmetric.pkl')#io.append_to_stem(args.dataset_pth, f'smiles_similarities'))
                model = ContrastiveHead(
                    args.pre_trained_pth,
                    args.lr,
                    args.weight_decay,
                    triplet_loss_margin=args.triplet_loss_margin,
                )
            elif args.train_objective == "mol_props":
                mol_props_calc = dataset.prop_calc
                model = RegressionHead(
                    backbone,
                    args.lr,
                    args.weight_decay,
                    sigmoid=False,
                    out_dim=len(mol_props_calc),
                    mol_props_calc=mol_props_calc,
                    head_depth=args.head_depth,
                    dropout=args.dropout,
                )

        # Set float64 weights
        if args.train_precision == 64:
            model = model.double()

        # ——— NeptuneLogger setup ———
        if not args.no_neptune:
            if cv:
                # for CV: re-name each fold run and carry over the “group” tag
                neptune_run = init_neptune(
                    tags=[*args.neptune_tags, args.run_name],  # include group tag
                    mode=args.neptune_mode,
                    name=f"{args.run_name} [fold_{i}]",  # per-fold name
                )
            else:
                # for single runs: just use the base name/tags
                neptune_run = init_neptune(
                    tags=args.neptune_tags,
                    mode=args.neptune_mode,
                    # mode="offline",
                    name=args.run_name,
                )

            neptune_logger = NeptuneLogger(
                run=neptune_run,
                log_model_checkpoints=False,
            )
            # push your full argparse config under training/hyperparams
            neptune_logger.log_hyperparams(fully_sanitize(vars(args)))
        else:
            neptune_logger = None

        # ——— WandbLogger setup ———
        if not args.no_wandb:
            assert (
                "WANDB_API_KEY" in os.environ
            ), "WANDB_API_KEY must be set in the environment variables."
            if cv:
                wandb.init(
                    reinit=True,
                    project=args.project_name,
                    name=f"{args.run_name} [fold_{i}]",
                    config=args,
                    group=args.run_name,
                    entity=args.wandb_entity_name,
                )
                wandb_logger = WandbLogger()
            else:
                wandb_logger = WandbLogger(
                    project=args.project_name,
                    name=args.run_name,
                    config=args,
                    entity=args.wandb_entity_name,
                )
        else:
            wandb_logger = None

        # Define trainer callbacks (TODO: understand the behavior of find_unused_parameters)
        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            pl.callbacks.ModelCheckpoint(
                monitor="Train loss",
                save_top_k=args.save_top_k,
                mode="min",
                dirpath=run_dir,
                save_last=True,
                every_n_train_steps=1000,
            ),
        ]

        if args.train_regime == "pre-training":
            # Define SSL probling callback
            if args.ssl_probing_dataset_pths:
                for pth in args.ssl_probing_dataset_pths:
                    df = pd.read_pickle(pth)
                    probing_data_module = du.SplittedDataModule(
                        du.AnnotatedSpectraDataset(
                            df["MSnSpectrum"].tolist(),
                            label="fp_maccs_166",
                            spec_preproc=spec_preproc,
                            dformat=args.dformat,
                            return_smiles=args.store_probing_pred,
                        ),
                        split_mask=df["val"] if "val" in df.columns else df["fold"],
                        batch_size=64,
                    )
                    callbacks.append(
                        du.SSLProbingValidation(
                            probing_data_module,
                            n_hidden_layers=args.ssl_probing_depth,
                            prefix=pth.stem,
                            save_fps_dir=args.gains_dir / "probing"
                            if args.store_probing_pred
                            else None,
                        )
                    )

        # Define trainer
        strategy = (
            pl.strategies.DDPStrategy(
                find_unused_parameters=False
                if args.train_regime == "pre-training"
                else True
            )
            if not cv
            else None
        )

        # ——— Combine both loggers for the Trainer ———
        loggers = [lg for lg in (wandb_logger, neptune_logger) if lg]

        trainer = pl.Trainer(
            strategy=strategy,
            max_epochs=args.max_epochs,
            logger=loggers or False,
            accelerator=device,
            devices=args.num_devices,
            log_every_n_steps=args.log_every_n_steps,
            precision=args.train_precision,
            overfit_batches=args.overfit_batches,
            callbacks=callbacks,
            num_sanity_val_steps=0,
            use_distributed_sampler=args.num_devices > 1,
            val_check_interval=None if args.no_val else args.val_check_interval,
            limit_val_batches=0 if args.no_val else None,
        )

        if not args.no_wandb and trainer.global_rank == 0:
            # Watch model on the main process
            wandb_logger.watch(model, log_graph=False)
            wandb_logger.experiment.config.update(
                {
                    "num_params": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    )
                }
            )

        trainer.validate(
            model,
            dataloaders=[l for l in [data_module.val_dataloader()] if l is not None],
        )

        trainer.fit(
            model,
            train_dataloaders=data_module.train_dataloader(),
            val_dataloaders=[
                l for l in [data_module.val_dataloader()] if l is not None
            ],
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
