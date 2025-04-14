import torch

# import torch.nn
from torch.utils.data import random_split, Dataset, DataLoader  # noqa: F401
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from dataset import LanguagePairDataset, generate_causal_mask
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from config import fetch_configuration, construct_model_path, latest_model_path
from model import build
from pathlib import Path
from tqdm import tqdm
import warnings
import os


def build_or_load_tokenizer(config, dataset, language):
    # config["tokenizer_path"] = "..tokenizer/tokenizer_{0}.json"
    tokenizer_path = Path(config["tokenizer_path"].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["<unk>", "<pad>", "<sos>", "<eos>"], min_frequency=2
        )
        sentences = (item["translation"][language] for item in dataset)
        tokenizer.train_from_iterator(sentences, trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def prepare_dataset(config):
    raw_dataset = load_dataset(
        "opus_books",
        f"{config['lang_src']}-{config['lang_tgt']}",
        split="train",
    )

    tokenizer_src = build_or_load_tokenizer(config, raw_dataset, config["lang_src"])
    tokenizer_tgt = build_or_load_tokenizer(config, raw_dataset, config["lang_tgt"])

    # 90/10 split for train/val
    train_size = int(0.9 * len(raw_dataset))
    val_size = len(raw_dataset) - train_size
    train_set_raw, val_set_raw = random_split(raw_dataset, [train_size, val_size])

    # Prepare the final train and val sets
    train_set = LanguagePairDataset(
        train_set_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_set = LanguagePairDataset(
        val_set_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    max_src_len, max_tgt_len = calculate_max_lengths(
        raw_dataset, tokenizer_src, tokenizer_tgt, config
    )
    print("Max length of source sentences: {}".format(max_src_len))
    print("Max length of target sentences: {}".format(max_tgt_len))

    # Dataloader
    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True)

    return train_loader, val_loader, tokenizer_src, tokenizer_tgt


def calculate_max_lengths(raw_ds, tokenizer_src, tokenizer_tgt, config):
    max_src_len = 0
    max_tgt_len = 0
    for it in raw_ds:
        src_len = len(tokenizer_src.encode(it["translation"][config["lang_src"]]).ids)
        tgt_len = len(tokenizer_tgt.encode(it["translation"][config["lang_tgt"]]).ids)
        max_src_len = max(max_src_len, src_len)
        max_tgt_len = max(max_tgt_len, tgt_len)
    return max_src_len, max_tgt_len


def build_transformer_model(config, src_vocab_size, tgt_vocab_size):
    model = build(
        src_vocab_size,
        tgt_vocab_size,
        config["seq_len"],
        config["seq_len"],
        config["hidden_size"],
        config["attention_dropout"],
        config["num_attention_heads"],
        config["num_hidden_layers"],
        config["intermediate_size"],
    )
    return model


def train_model(config):
    # Switch to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in use: {device}")
    # Weights dir
    Path(config["model_dir"]).mkdir(parents=True, exist_ok=True)

    # Create Dataloader, Model, Optimizer, Loss Function
    train_loader, val_loader, tokenizer_src, tokenizer_tgt = prepare_dataset(config)
    transformer = build_transformer_model(
        config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()
    ).to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=config["lr"], eps=1e-9)
    loss_func = torch.nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("<pad>"), label_smoothing=0.1
    ).to(device)

    # Tensorboard
    logger = SummaryWriter(config["tensorboard_run_name"])

    # Preload model
    initial_epoch = 0
    global_step = 0
    preload = config["preload"]
    model_filename = (
        latest_model_path(config)
        if preload == "latest"
        else construct_model_path(config, preload)
        if preload
        else None
    )
    if model_filename:
        print("Preloading model {}".format(model_filename))
        state = torch.load(model_filename)
        transformer.load_state_dict(state["model_state_dict"])
        # update to the last completed epoch
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    # Training Loop
    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        transformer.train()
        batch_iter = tqdm(train_loader, desc=f"Epoch {epoch:02d}")
        for batch in batch_iter:
            # Load the batch on the device
            encoder_input = batch["encoder_input"].to(device)  # (bat,seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (bat,seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (bat,1,1,seq_len)
            causal_mask = batch["causal_mask"].to(device)  # (bat,1,seq_len,seq_len)
            true_label = batch["target"].to(device)  # (bat,seq_len)

            # Run tensors through the transformer
            encoder_output = transformer.encode(
                encoder_input, encoder_mask
            )  # (bat,seq_len,hidden_size)
            decoder_output = transformer.decode(
                encoder_output, encoder_mask, decoder_input, causal_mask
            )  # (bat,seq_len,hidden_size)
            linear_output = transformer.linear_proj(
                decoder_output
            )  # (bat,seq_len,tgt_vocab_size)

            # Loss calculation
            # view operation 1: (bat, seq_len, tgt_vocab_size] -> [(B * seq_len), tgt_vocab_size]
            # view operation 2: (bat,seq_len) -> (bat * seq_len)
            loss = loss_func(
                linear_output.view(-1, tokenizer_tgt.get_vocab_size()),
                true_label.view(-1),
            )

            # Update progress bar
            batch_iter.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss into tensorboard
            logger.add_scalar("training_loss", loss.item(), global_step=global_step)
            logger.flush()

            # optimize:
            loss.backward()  # backprop
            optimizer.step()  # weight update
            optimizer.zero_grad()  # clear grads at each iteration

            global_step += 1

        # Run a validation loop
        perform_validation(
            transformer,
            val_loader,
            tokenizer_src,
            tokenizer_tgt,
            config["seq_len"],
            global_step,
            lambda msg: batch_iter.write(msg),
            logger,
            device,
        )

        # Save the weights at each epoch
        model_filename = construct_model_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "model_state_dict": transformer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


def greedyDecoder(
    transformer,
    src,
    src_mask,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
):
    sos_tkn_id = tokenizer_tgt.token_to_id("<sos>")
    eos_tkn_id = tokenizer_tgt.token_to_id("<eos>")

    # Auto-regressively use precomputed encoder generated context-vectors to generate output tokens
    encoder_output = transformer.encode(src, src_mask)
    # decoder input for starting sequence generation
    decoder_seq = torch.full((1, 1), sos_tkn_id, dtype=src.dtype, device=device)

    while decoder_seq.size(1) < max_len:
        # Causal mask
        causal_mask = (
            generate_causal_mask(decoder_seq.size(1)).type_as(src_mask).to(device)
        )
        # Decoder Output
        output = transformer.decode(encoder_output, src_mask, decoder_seq, causal_mask)
        predictions = transformer.linear_proj(output[:, -1])
        next_token = predictions.argmax(1)
        decoder_seq = torch.cat([decoder_seq, next_token.unsqueeze(0)], dim=1).to(
            device
        )

        if next_token.item() == eos_tkn_id:
            break
    return decoder_seq.squeeze()


def perform_validation(
    transformer,
    val_set,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    global_step,
    print_msg,
    logger,
    device,
    num_examples=3,
):
    transformer.eval()
    samples_processed = 0
    # Control window dimensions
    # console_width = 80
    console_width = os.get_terminal_size().columns if os.isatty(1) else 80

    with torch.no_grad():
        for batch in val_set:
            if samples_processed == num_examples:
                break

            src = batch["encoder_input"].to(device)
            src_mask = batch["encoder_mask"].to(device)
            assert src.size(0) == 1, "Validation batch size must be 1."

            decoded_output = greedyDecoder(
                transformer,
                src,
                src_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
            )

            src_txt = batch["src_text"][0]
            true_txt = batch["tgt_text"][0]
            pred_txt = tokenizer_tgt.decode(decoded_output.detach().cpu().numpy())

            # Print the validation process
            print_msg("=" * console_width)
            print_msg(f"SOURCE: {src_txt}")
            print_msg(f"TRUE: {true_txt}")
            print_msg(f"PREDICTED: {pred_txt}")

            samples_processed += 1


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = fetch_configuration()
    train_model(config)
