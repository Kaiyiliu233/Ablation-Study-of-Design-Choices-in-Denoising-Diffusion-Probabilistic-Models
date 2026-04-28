import os, sys, argparse, glob
import torch
from torchvision.utils import save_image

from config import ALL_CONFIGS
from diffusion import Diffusion
from model import UNet
from train import resolve_experiment, apply_experiment_paths


def find_checkpoint(cfg, step=None):
    if step is not None:
        path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_name}_step{step}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path
    pattern = os.path.join(cfg.checkpoint_dir, f"{cfg.run_name}_step*.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"no checkpoints matching {pattern}")
    def get_step(p):
        return int(os.path.basename(p).split("_step")[-1].replace(".pt", ""))
    return max(files, key=get_step)


def build_model(cfg, device):
    return UNet(
        in_ch=cfg.in_ch, out_ch=cfg.out_ch,
        base_ch=cfg.base_ch, ch_mults=cfg.ch_mults,
        num_res_blocks=cfg.num_res_blocks,
        time_dim=cfg.time_dim, attn_at_bottom=cfg.attn_at_bottom,
    ).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run name")
    ap.add_argument("--step", type=int, default=None, help="checkpoint step")
    ap.add_argument("--n", type=int, default=10000, help="number of samples")
    ap.add_argument("--batch", type=int, default=256, help="samples per batch")
    ap.add_argument("--device", default=None, help="override cfg.device")
    ap.add_argument("--out", default=None, help="override output dir")
    args = ap.parse_args()

    if args.run not in ALL_CONFIGS:
        print(f"Unknown run: {args.run}. Choose from: {list(ALL_CONFIGS.keys())}")
        sys.exit(1)

    cfg = ALL_CONFIGS[args.run]
    apply_experiment_paths(cfg)
    device = args.device or cfg.device

    ckpt_path = find_checkpoint(cfg, args.step)
    print(f"loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    step = ckpt["step"]

    model = build_model(cfg, device)
    if "ema" in ckpt:
        print("using EMA weights")
        ema_state = ckpt["ema"]
        # ema_pytorch flattens keys as "ema_model.<...>" / "online_model.<...>"
        prefix = "ema_model."
        ema_weights = {k[len(prefix):]: v for k, v in ema_state.items() if k.startswith(prefix)}
        model.load_state_dict(ema_weights)
    else:
        model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = Diffusion(T=cfg.T, schedule=cfg.schedule, prediction=cfg.prediction, device=device)

    exp = resolve_experiment(cfg.run_name)
    out_dir = args.out or os.path.join("samples", exp, f"{cfg.run_name}_fid_step{step}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"saving to {out_dir}")

    n_done = 0
    batch_idx = 0
    while n_done < args.n:
        bs = min(args.batch, args.n - n_done)
        x = diffusion.sample(model, shape=(bs, cfg.in_ch, 32, 32))
        x = (x.clamp(-1, 1) + 1) / 2

        for i in range(bs):
            save_image(x[i], os.path.join(out_dir, f"{n_done + i:05d}.png"))
        n_done += bs
        batch_idx += 1
        print(f"  [{n_done}/{args.n}] batch {batch_idx}")

if __name__ == "__main__":
    main()
