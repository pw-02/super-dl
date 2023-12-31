# Imports other packages.
import configargparse
import torchvision.models as models

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

def parse_args(config_file):
    """Parses command line and config file arguments."""
    parser = configargparse.ArgumentParser(
        description='Testing out serverless data loader on CPU or GPU',
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=False,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        default_config_files=[config_file])

    parser = add_input_args(parser)
    args = parser.parse_args()
    return args

def add_input_args(parser):
    """Adds arguments not handled by Trainer or model."""
    parser.add(
        "--gprc_server_address",
        default='localhost:50051',
        help="port for gprc server.",
    )
    parser.add(
        "--arch",
        default='resnet18',
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet18)"
        )
    
    parser.add(
        "--weight_decay",
        type=float,
        help="Weight decay factor.",
    )

    parser.add(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum.",
    )

    parser.add(
        "--lr",
        type=float,
        default=0.1,
        help="initial learning rate",
    )

    parser.add(
        "--save_dir",
        default="out",
        help="Directory to save images and checkpoints; overrides PT dir.",
    )

    parser.add(
        "--data_dir",
        default="data",
       
    )
    parser.add(
        "--dataset_name",
    )

    parser.add(
        "--log_dir",
        default="logs",
       
    )
    parser.add(
        "--task",
        choices=["train", "test", "infer"],
        help="Mode to run the model in.",
    )  
    parser.add(
        "--sampler",
        action="store_true",
        help="Whether to use a balanced random sampler in DataLoader.",
    )
    parser.add(
        "--always_save_checkpoint",
        action="store_true",
    )

    parser.add(
        "--num_workers",
        type=int,
        help="Number of workers in DataLoader.",
    )
    parser.add(
        "--gpus",
        type=int,
    )
    parser.add(
        "--lr_step_size",
        type=int,
        help="How many epochs to run before dropping the learning rate.",
    )

    parser.add(
        "--pin_memory",
        action="store_true",
        help=(
            "pin_memory."
        ),
    )
    parser.add(
        "--max_epochs",
        type=int,
        help="Max number of epochs.",
    )
    parser.add(
        "--batch_size",
        type=int,
        help="Number of images per batch.",
    )
    parser.add(
        "--log_interval",
        type=int,
        default=10,
        help="how many batches to wait before logging training status",
    )
    parser.add(
        "--source_system",
        default="s3",
       
    )
    parser.add(
        "--cache_host",
        default="10.0.1.197:6379",
    )

    parser.add(
        "--dry_run",
        action="store_true",
    )
    parser.add(
        "--data_profile",
        action="store_true",
    )
    #parser.add(
    #    "--full_epoch",
    #    action="store_true",
    #)
    parser.add(
        "--max_minibatches_per_epoch",
        type=int,
        default=None,
    )
    parser.add(
        "--train_only",
        action="store_true",
    )

    parser.add(
        "--use_super",
        action="store_true",
    )
    return parser