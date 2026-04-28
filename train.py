import os, sys, csv, time, glob
import logging
import torch
from ema_pytorch import EMA
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from config import ALL_CONFIGS, EXPERIMENTS
from diffusion import Diffusion
from model import UNet


def resolve_experiment(run_name):
    for exp_name, run_names in EXPERIMENTS.items():
        if run_name in run_names:
            return exp_name
    return run_name

def apply_experiment_paths(cfg):
    exp = resolve_experiment(cfg.run_name)
    if os.path.basename(cfg.checkpoint_dir) != exp:
        cfg.checkpoint_dir = os.path.join(cfg.checkpoint_dir, exp)
    if os.path.basename(cfg.sample_dir) != exp:
        cfg.sample_dir = os.path.join(cfg.sample_dir, exp)


def setup_logging(cfg):
    logger = logging.getLogger("DDPM")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(cfg.checkpoint_dir, f"{cfg.run_name}.log"), mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def get_dataloader(cfg):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)


def find_latest_checkpoint(cfg):
    pattern = os.path.join(cfg.checkpoint_dir, f"{cfg.run_name}_step*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    def get_step(p):
        return int(os.path.basename(p).split("_step")[-1].replace(".pt", ""))
    return max(files, key=get_step)

def load_checkpoint(path, model, optimizer, device, ema=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if ema is not None and "ema" in ckpt:
        ema.load_state_dict(ckpt["ema"])
    return ckpt["step"]

def save_checkpoint(cfg, model, optimizer, step, ema=None):
    data = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
    }
    if ema is not None:
        data["ema"] = ema.state_dict()
    torch.save(data, os.path.join(cfg.checkpoint_dir, f"{cfg.run_name}_step{step}.pt"))


def setup_csv_logger(cfg):
    log_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_name}_loss.csv")
    exists = os.path.exists(log_path)
    f = open(log_path, "a", newline="")
    w = csv.writer(f)
    if not exists:
        w.writerow(["step", "epoch", "loss", "elapsed_sec"])
    return w, f


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def evaluate_and_save(cfg, diffusion, model, optimizer, step, logger, tag=None, ema=None):
    eval_model = ema.ema_model if ema is not None else model
    eval_model.eval()
    samples = diffusion.sample(eval_model, shape=(64, cfg.in_ch, 32, 32))
    samples = (samples.clamp(-1, 1) + 1) / 2
    suffix = tag if tag is not None else f"step{step}"
    save_image(samples, os.path.join(cfg.sample_dir, f"{cfg.run_name}_{suffix}.png"), nrow=8)
    save_checkpoint(cfg, model, optimizer, step, ema=ema)
    logger.info(f"  checkpoint + samples saved @ step {step}")
    model.train()


def train(cfg):
    apply_experiment_paths(cfg)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.sample_dir, exist_ok=True)
    logger = setup_logging(cfg)

    diffusion = Diffusion(T=cfg.T, schedule=cfg.schedule, prediction=cfg.prediction, device=cfg.device)
    model = UNet(
        in_ch=cfg.in_ch, out_ch=cfg.out_ch,
        base_ch=cfg.base_ch, ch_mults=cfg.ch_mults,
        num_res_blocks=cfg.num_res_blocks,
        time_dim=cfg.time_dim, attn_at_bottom=cfg.attn_at_bottom,
    ).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ema = EMA(model, beta=cfg.ema_decay, update_every=1, update_after_step=0) if cfg.use_ema else None
    n_params = sum(p.numel() for p in model.parameters())

    logger.info("DDPM Training")
    logger.info(f"Run:{cfg.run_name}")
    logger.info(f"Schedule:{cfg.schedule}")
    logger.info(f"Prediction:{cfg.prediction}")
    logger.info(f"EMA:{cfg.use_ema}" + (f" (decay={cfg.ema_decay})" if cfg.use_ema else ""))
    logger.info(f"Base ch:{cfg.base_ch}")
    logger.info(f"Parameters:{n_params:,}")
    logger.info(f"Batch size:{cfg.batch_size}")
    logger.info(f"LR:{cfg.lr}")
    logger.info(f"Steps:{cfg.total_steps:,}")
    logger.info(f"Device:{cfg.device}")

    # resume if possible
    start_step = 0
    latest_ckpt = find_latest_checkpoint(cfg)
    if latest_ckpt is not None:
        start_step = load_checkpoint(latest_ckpt, model, optimizer, cfg.device, ema=ema)
        logger.info(f"Resumed from checkpoint at step {start_step}")
    else:
        logger.info("Starting from scratch")

    if start_step >= cfg.total_steps:
        logger.info("Training already complete.")
        return

    dataloader = get_dataloader(cfg)
    steps_per_epoch = len(dataloader)
    total_epochs = cfg.total_steps / steps_per_epoch
    logger.info(f"Dataset: {len(dataloader.dataset):,} images")
    logger.info(f"Steps/epoch: {steps_per_epoch}")
    logger.info(f"Est epochs: {total_epochs:.1f}")


    csv_writer, csv_file = setup_csv_logger(cfg)

    running_loss = 0.0
    log_interval = 100
    t0 = time.time()
    step = start_step
    epoch = step // steps_per_epoch

    model.train()
    while step < cfg.total_steps:
        epoch += 1
        logger.info(f"Epoch {epoch} starting (step {step}/{cfg.total_steps})")

        for x0, _ in dataloader:
            if step >= cfg.total_steps:
                break

            x0 = x0.to(cfg.device)
            loss = diffusion.training_step(model, x0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update()

            step += 1
            running_loss += loss.item()

            if step % log_interval == 0:
                avg_loss = running_loss / log_interval
                elapsed = time.time() - t0
                steps_done = step - start_step
                sps = steps_done / elapsed
                eta = (cfg.total_steps - step) / sps if sps > 0 else 0
                pct = step / cfg.total_steps * 100
                logger.info(
                    f"  [{pct:5.1f}%] "
                    f"epoch {epoch} | "
                    f"step {step:>7d}/{cfg.total_steps} | "
                    f"loss {avg_loss:.4f} | "
                    f"{sps:.1f} it/s | "
                    f"elapsed {format_time(elapsed)} | "
                    f"eta {format_time(eta)}"
                )
                csv_writer.writerow([step, epoch, f"{avg_loss:.6f}", f"{elapsed:.1f}"])
                csv_file.flush()
                running_loss = 0.0

            if step % cfg.eval_interval == 0:
                logger.info(f"generating sample grid @ step {step}")
                evaluate_and_save(cfg, diffusion, model, optimizer, step, logger, ema=ema)

        logger.info(f"Epoch {epoch} complete (step {step}/{cfg.total_steps})")

    if step % cfg.eval_interval != 0:
        logger.info("Generating final samples")
        evaluate_and_save(cfg, diffusion, model, optimizer, step, logger, tag="final", ema=ema)

    csv_file.close()
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"Training complete. Total time: {format_time(elapsed)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "debug"
    if name in EXPERIMENTS:
        for run_name in EXPERIMENTS[name]:
            print(f"\n Running {run_name} ({name})\n")
            train(ALL_CONFIGS[run_name])
    elif name in ALL_CONFIGS:
        train(ALL_CONFIGS[name])
    else:
        valid = list(EXPERIMENTS.keys()) + list(ALL_CONFIGS.keys())
        print(f"Unknown name: {name}. Choose from: {valid}")
        sys.exit(1)
