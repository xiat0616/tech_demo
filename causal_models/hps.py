# import copy

# from traitlets import default

HPARAMS_REGISTRY = {}

def value_for_none(v):
    if not v or v=="None":
        return None
    return float(v)
    


class Hparams:
    def update(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)

mimic256_64 = Hparams()
mimic256_64.batch_size = 16
mimic256_64.input_channels = 1
mimic256_64.lr = 1e-3
mimic256_64.wd = 1e-3
mimic256_64.lr_warmup_steps = 100
mimic256_64.bottleneck = 4
mimic256_64.cond_prior = True
mimic256_64.z_max_res = 128
mimic256_64.z_dim = [48, 30, 24, 18, 12, 6, 1]
mimic256_64.input_res = [256,64]
mimic256_64.enc_arch = "256_64b2d2_2,128_32b4d2_2,64_16b9d2_2,32_8b4d2_2,16_4b4d2_2,8_2b9d8_2,1_1b5"
mimic256_64.dec_arch = "1_1b5,8_2b10,16_4b15,32_8b15,64_16b10,128_32b5,256_64b3"
mimic256_64.widths = [32, 64, 96, 128, 160, 192, 512]
mimic256_64.parents_x = ["age", "race", "sex", "finding"]
mimic256_64.context_dim = 6
mimic256_64.bias_max_res = 128
mimic256_64.embd_dim = 32
HPARAMS_REGISTRY["mimic256_64"] = mimic256_64


mimic256_64_with_seg = Hparams()
mimic256_64_with_seg.batch_size = 16
mimic256_64_with_seg.input_channels = 1
mimic256_64_with_seg.lr = 1e-3
mimic256_64_with_seg.wd = 1e-3
mimic256_64_with_seg.lr_warmup_steps = 100
mimic256_64_with_seg.bottleneck = 4
mimic256_64_with_seg.cond_prior = True
mimic256_64_with_seg.z_max_res = 128
mimic256_64_with_seg.z_dim = [48, 30, 24, 18, 12, 6, 1]
mimic256_64_with_seg.input_res = [256,64]
mimic256_64_with_seg.enc_arch = "256_64b2d2_2,128_32b4d2_2,64_16b9d2_2,32_8b4d2_2,16_4b4d2_2,8_2b9d8_2,1_1b5"
mimic256_64_with_seg.dec_arch = "1_1b5,8_2b10,16_4b15,32_8b15,64_16b10,128_32b5,256_64b3"
mimic256_64_with_seg.widths = [32, 64, 96, 128, 160, 192, 512]
mimic256_64_with_seg.parents_x = ["Left-Lung_volume", "Right-Lung_volume", "Heart_volume"]
mimic256_64_with_seg.context_dim = 3
mimic256_64_with_seg.bias_max_res = 128
mimic256_64_with_seg.embd_dim = 32
HPARAMS_REGISTRY["mimic256_64_with_seg"] = mimic256_64_with_seg


mimic224_224_with_seg = Hparams()
mimic224_224_with_seg.batch_size = 16
mimic224_224_with_seg.input_channels = 1
mimic224_224_with_seg.lr = 1e-3
mimic224_224_with_seg.wd = 1e-3
mimic224_224_with_seg.lr_warmup_steps = 100
mimic224_224_with_seg.bottleneck = 4
mimic224_224_with_seg.cond_prior = True
mimic224_224_with_seg.z_max_res = 128
mimic224_224_with_seg.z_dim = [48, 30, 24, 18, 12, 6, 1]
mimic224_224_with_seg.input_res = [224,224]
mimic224_224_with_seg.enc_arch = "224_224b2d2_2,112_112b4d2_2,56_56b9d2_2,28_28b4d2_2,14_14b4d2_2,7_7b9d7_7,1_1b5"
mimic224_224_with_seg.dec_arch = "1_1b5,7_7b10,14_14b15,28_28b15,56_56b10,112_112b5,224_224b3"
mimic224_224_with_seg.widths = [32, 64, 96, 128, 160, 192, 512]
mimic224_224_with_seg.parents_x = ["Left-Lung_volume", "Right-Lung_volume", "Heart_volume"]
mimic224_224_with_seg.context_dim = 3
mimic224_224_with_seg.bias_max_res = 128
mimic224_224_with_seg.embd_dim = 32
HPARAMS_REGISTRY["mimic224_224_with_seg"] = mimic224_224_with_seg

# For RSNA dataset
rsna224_224 = Hparams()
rsna224_224.batch_size = 16
rsna224_224.input_channels = 1
rsna224_224.lr = 1e-3
rsna224_224.wd = 1e-3
rsna224_224.lr_warmup_steps = 100
rsna224_224.bottleneck = 4
rsna224_224.cond_prior = True
rsna224_224.z_max_res = 128
rsna224_224.z_dim = [48, 30, 24, 18, 12, 6, 1]
rsna224_224.input_res = [224,224]
rsna224_224.enc_arch = "224_224b2d2_2,112_112b4d2_2,56_56b9d2_2,28_28b4d2_2,14_14b4d2_2,7_7b9d7_7,1_1b5"
rsna224_224.dec_arch = "1_1b5,7_7b10,14_14b15,28_28b15,56_56b10,112_112b5,224_224b3"
rsna224_224.widths = [32, 64, 96, 128, 160, 192, 512]
rsna224_224.parents_x = ["age", "sex", "finding"]
rsna224_224.context_dim = 3
rsna224_224.bias_max_res = 128
rsna224_224.embd_dim = 32
HPARAMS_REGISTRY["rsna224_224"] = rsna224_224

rsna224_224_with_seg = Hparams()
rsna224_224_with_seg.batch_size = 16
rsna224_224_with_seg.input_channels = 1
rsna224_224_with_seg.lr = 1e-3
rsna224_224_with_seg.wd = 1e-3
rsna224_224_with_seg.lr_warmup_steps = 100
rsna224_224_with_seg.bottleneck = 4
rsna224_224_with_seg.cond_prior = True
rsna224_224_with_seg.z_max_res = 128
rsna224_224_with_seg.z_dim = [48, 30, 24, 18, 12, 6, 1]
rsna224_224_with_seg.input_res = [224,224]
rsna224_224_with_seg.enc_arch = "224_224b2d2_2,112_112b4d2_2,56_56b9d2_2,28_28b4d2_2,14_14b4d2_2,7_7b9d7_7,1_1b5"
rsna224_224_with_seg.dec_arch = "1_1b5,7_7b10,14_14b15,28_28b15,56_56b10,112_112b5,224_224b3"
rsna224_224_with_seg.widths = [32, 64, 96, 128, 160, 192, 512]
rsna224_224_with_seg.parents_x = ["Left-Lung_volume", "Right-Lung_volume", "Heart_volume"]
rsna224_224_with_seg.context_dim = 3
rsna224_224_with_seg.bias_max_res = 128
rsna224_224_with_seg.embd_dim = 32
HPARAMS_REGISTRY["rsna224_224_with_seg"] = rsna224_224_with_seg

# For PAD_CHEST dataset
padchest224_224 = Hparams()
padchest224_224.batch_size = 16
padchest224_224.input_channels = 1
padchest224_224.lr = 1e-3
padchest224_224.wd = 1e-3
padchest224_224.lr_warmup_steps = 100
padchest224_224.bottleneck = 4
padchest224_224.cond_prior = True
padchest224_224.z_max_res = 128
padchest224_224.z_dim = [48, 30, 24, 18, 12, 6, 1]
padchest224_224.input_res = [224,224]
padchest224_224.enc_arch = "224_224b2d2_2,112_112b4d2_2,56_56b9d2_2,28_28b4d2_2,14_14b4d2_2,7_7b9d7_7,1_1b5"
padchest224_224.dec_arch = "1_1b5,7_7b10,14_14b15,28_28b15,56_56b10,112_112b5,224_224b3"
padchest224_224.widths = [32, 64, 96, 128, 160, 192, 512]
padchest224_224.parents_x = ["age", "sex"]
padchest224_224.context_dim = 3
padchest224_224.bias_max_res = 128
padchest224_224.embd_dim = 32
HPARAMS_REGISTRY["padchest224_224"] = padchest224_224

def setup_hparams(parser):
    hparams = Hparams()
    args = parser.parse_known_args()[0]
    valid_args = set(args.__dict__.keys())
    hparams_dict = HPARAMS_REGISTRY[args.hps].__dict__
    for k in hparams_dict.keys():
        if k not in valid_args:
            raise ValueError(f"{k} not in default args") 
    parser.set_defaults(**hparams_dict)
    hparams.update(parser.parse_known_args()[0].__dict__)
    return hparams


def add_arguments(parser):
    parser.add_argument("--batch_size", help="The batch size.", type=int, default=16)
    parser.add_argument("--exp_name", help="Experiment name.", type=str, default="")
    parser.add_argument(
        "--data_dir", help="Data directory to load form.", type=str, default=""
    )
    parser.add_argument("--hps", help="hyperparam set.", type=str, default="ukbb64")
    parser.add_argument(
        "--resume", help="Path to load checkpoint.", type=str, default=""
    )
    parser.add_argument("--seed", help="Set random seed.", type=int, default=7)
    parser.add_argument(
        "--deterministic",
        help="Toggle cudNN determinism.",
        action="store_true",
        default=False,
    )
    # training
    parser.add_argument("--epochs", help="Training epochs.", type=int, default=5000)
    parser.add_argument("--lr", help="Learning rate.", type=float, default=1e-3)
    parser.add_argument(
        "--lr_warmup_steps", help="lr warmup steps.", type=int, default=100
    )
    parser.add_argument("--lambda_cr", help="Lambda for CR loss.", type=value_for_none)
    parser.add_argument("--wd", help="Weight decay penalty.", type=float, default=0.01)
    parser.add_argument(
        "--betas",
        help="Adam beta parameters.",
        nargs="+",
        type=float,
        default=[0.9, 0.9],
    )
    parser.add_argument(
        "--ema_rate", help="Exp. moving avg. model rate.", type=float, default=0.999
    )
    parser.add_argument(
        "--input_res", help="Input image crop resolution.", type=int, default=64
    )
    parser.add_argument(
        "--input_channels", help="Input image num channels.", type=int, default=1
    )
    parser.add_argument(
        "--grad_clip", help="Gradient clipping value.", type=float, default=350
    )
    parser.add_argument(
        "--grad_skip", help="Skip update grad norm threshold.", type=float, default=500
    )
    parser.add_argument(
        "--accu_steps", help="Gradient accumulation steps.", type=int, default=1
    )
    parser.add_argument(
        "--beta", help="Max KL beta penalty weight.", type=float, default=1.0
    )
    parser.add_argument(
        "--beta_warmup_steps", help="KL beta penalty warmup steps.", type=int, default=0
    )
    parser.add_argument(
        "--kl_free_bits", help="KL min free bits constraint.", type=float, default=0.0
    )
    parser.add_argument(
        "--viz_freq", help="Steps per visualisation.", type=int, default=5000
    )
    parser.add_argument(
        "--eval_freq", help="Train epochs per validation.", type=int, default=5
    )
    parser.add_argument(
        "--enc_arch",
        help="Encoder architecture config.",
        type=str,
        default="64b1d2,32b1d2,16b1d2,8b1d8,1b2",
    )
    parser.add_argument(
        "--dec_arch",
        help="Decoder architecture config.",
        type=str,
        default="1b2,8b2,16b2,32b2,64b2",
    )
    parser.add_argument(
        "--cond_prior",
        help="Use a conditional prior.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--widths",
        help="Number of channels.",
        nargs="+",
        type=int,
        default=[16, 32, 48, 64, 128],
    )
    parser.add_argument(
        "--bottleneck", help="Bottleneck width factor.", type=int, default=4
    )
    parser.add_argument(
        "--z_dim", help="Numver of latent channel dims.", type=int, default=16
    )
    parser.add_argument(
        "--z_max_res",
        help="Max resolution of stochastic z layers.",
        type=int,
        default=192,
    )
    parser.add_argument(
        "--bias_max_res",
        help="Learned bias param max resolution.",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--x_like",
        help="x likelihood: {fixed/shared/diag}_{gauss/dgauss}.",
        type=str,
        default="diag_dgauss",
    )
    parser.add_argument(
        "--std_init",
        help="Initial std for x scale. 0 is random.",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--parents_x",
        help="Parents of x to condition on.",
        nargs="+",
        default=["mri_seq", "brain_volume", "ventricle_volume", "sex"],
    )
    parser.add_argument(
        "--concat_pa",
        help="Whether to concatenate parents_x.",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--embd_dim",
        help="Embedding dim",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--context_dim",
        help="Num context variables conditioned on.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--context_norm",
        help='Conditioning normalisation {"[-1,1]"/"[0,1]"/log_standard}.',
        type=str,
        default="log_standard",
    )
    parser.add_argument(
        "--q_correction",
        help="Use posterior correction.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--cond_drop",
        help="Use counterfactual dropout",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--p_dropout",
        help="Block dropout",
        type=float,
        default=0.1,
    )

    return parser