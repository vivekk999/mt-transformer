from pathlib import Path


def fetch_configuration():
    """
    Returns:
    - dict: A dictionary with all configuration settings.
    """
    return {
        "hidden_size": 512,
        "seq_len": 350,
        "attention_dropout": 0.1,
        "intermediate_size": 2048,
        "num_attention_heads": 8,
        "num_hidden_layers": 6,
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 1e-4,
        "data_source": "opus_books",
        "lang_src": "en",
        "lang_tgt": "es",
        "model_dir": "weights",
        "model_name_prefix": "tmodel_",
        "preload": None,
        "tokenizer_path": "tokenizer_{0}.json",
        "tensorboard_run_name": "runs/tmodel",
    }


def construct_model_path(config, epoch: str):
    """
    Constructs a file path for storing or retrieving a model based on the epoch.
    """
    # directory_path = Path(".") / f"{config['data_source']}_{config['model_dir']}"
    directory_path = f"{config['model_dir']}"
    filename = f"{config['model_name_prefix']}{epoch}.pt"
    return str(Path('.') / directory_path / filename)


def latest_model_path(config):
    # model_folder = f"{config['data_source']}_{config['model_dir']}"
    model_folder = f"{config['model_dir']}"
    filename = f"{config['model_name_prefix']}*"
    weights_files = list(Path(model_folder).glob(filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
