from dataclasses import dataclass, field

@dataclass
class Config:
    T: int = 1000
    schedule: str = "cosine"
    prediction: str = "epsilon"

    # U-Net
    in_ch: int = 3
    out_ch: int = 3
    base_ch: int = 64
    ch_mults: tuple = (1, 2, 4)
    num_res_blocks: int = 2
    time_dim: int = 256
    attn_at_bottom: bool = True

    # Training
    batch_size: int = 128
    lr: float = 2e-4
    total_steps: int = 600_000
    device: str = "cuda"

    # EMA
    use_ema: bool = False
    ema_decay: float = 0.9999

    # Eval
    eval_interval: int = 25_000
    num_eval_samples: int = 10_000

    # Paths
    checkpoint_dir: str = "checkpoints"
    sample_dir: str = "samples"
    run_name: str = "baseline"


# debug config
DEBUG = Config(
    base_ch=16,
    ch_mults=(1, 2),
    num_res_blocks=1,
    T=100,
    batch_size=8,
    total_steps=500,
    eval_interval=250,
    run_name="debug",
)


# experiment 1: noise schedule
SCHEDULE_LINEAR = Config(schedule="linear",  prediction="epsilon", use_ema=False, run_name="exp1_schedule_linear")
SCHEDULE_COSINE = Config(schedule="cosine",  prediction="epsilon", use_ema=False, run_name="exp1_schedule_cosine")
SCHEDULE_SIGMOID= Config(schedule="sigmoid", prediction="epsilon", use_ema=False, run_name="exp1_schedule_sigmoid")

# experiment 2: EMA
EMA_OFF         = Config(schedule="cosine", prediction="epsilon", use_ema=False, run_name="exp2_ema_off")
EMA_ON          = Config(schedule="cosine", prediction="epsilon", use_ema=True,  run_name="exp2_ema_on")
EMA_ON_LINEAR   = Config(schedule="linear", prediction="epsilon", use_ema=True,  run_name="exp2_ema_on_linear")

# experiment 3: loss target
PREDICT_EPSILON = Config(schedule="cosine", prediction="epsilon", use_ema=True, run_name="exp3_predict_epsilon")
PREDICT_X0      = Config(schedule="cosine", prediction="x0",      use_ema=True, run_name="exp3_predict_x0")

# experiment 4: network size
SIZE_SMALL = Config(
    schedule="cosine", prediction="epsilon", use_ema=True,
    base_ch=32, num_res_blocks=1, batch_size=128,
    run_name="exp4_size_small",
)
SIZE_MEDIUM = Config(
    schedule="cosine", prediction="epsilon", use_ema=True,
    base_ch=64, num_res_blocks=2, batch_size=128,
    run_name="exp4_size_medium",
)
SIZE_LARGE = Config(
    schedule="cosine", prediction="epsilon", use_ema=True,
    base_ch=128, num_res_blocks=2, batch_size=64,  # smaller batch so it fits
    run_name="exp4_size_large",
)


ALL_CONFIGS = {
    "debug": DEBUG,
    # exp1
    "exp1_schedule_linear":  SCHEDULE_LINEAR,
    "exp1_schedule_cosine":  SCHEDULE_COSINE,
    "exp1_schedule_sigmoid": SCHEDULE_SIGMOID,
    # exp2
    "exp2_ema_off":       EMA_OFF,
    "exp2_ema_on":        EMA_ON,
    "exp2_ema_on_linear": EMA_ON_LINEAR,
    # exp3
    "exp3_predict_epsilon": PREDICT_EPSILON,
    "exp3_predict_x0":      PREDICT_X0,
    # exp4
    "exp4_size_small":  SIZE_SMALL,
    "exp4_size_medium": SIZE_MEDIUM,
    "exp4_size_large":  SIZE_LARGE,
}

EXPERIMENTS = {
    "exp1_schedule": [
        "exp1_schedule_linear",
        "exp1_schedule_cosine",
        "exp1_schedule_sigmoid",
    ],
    "exp2_ema": [
        "exp2_ema_off",
        "exp2_ema_on",
        "exp2_ema_on_linear",
    ],
    "exp3_prediction": [
        "exp3_predict_epsilon",
        "exp3_predict_x0",
    ],
    "exp4_size": [
        "exp4_size_small",
        "exp4_size_medium",
        "exp4_size_large",
    ],
}
