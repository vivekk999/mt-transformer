#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune a translation model using custom Transformer implementation
"""

import os
import sys
import time
import argparse
import threading
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import evaluate
from tqdm import tqdm

# Import custom model and components
from model import build
from dataset import LanguagePairDataset, generate_causal_mask
from config import fetch_configuration, construct_model_path

try:
    import pynvml  # Import for GPU usage logging
except ImportError:
    pynvml = None


class EarlyStoppingCallback:
    """
    Implements early stopping for model training
    """
    def __init__(self, patience=3):
        self.patience = patience
        self.best_score = float('inf')
        self.wait = 0
        self.should_stop = False
        
    def check(self, eval_loss):
        if eval_loss < self.best_score:
            self.best_score = eval_loss
            self.wait = 0
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                return True
            return False


def gpu_monitor(log_file_path):
    """
    Monitors GPU usage and logs to a file
    """
    if pynvml is None:
        print("pynvml not installed, GPU monitoring disabled")
        return
    
    pynvml.nvmlInit()  # Initialize NVML
    device_count = pynvml.nvmlDeviceGetCount()
    with open(log_file_path, 'w') as log_file:
        log_file.write("time,gpu_id,memory_used_MB,utilization_percent,temperature_C,power_draw_W\n")
        try:
            while True:
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
                    log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{i},{memory_info.used // (1024 * 1024)},{utilization.gpu},{temperature},{power_draw}\n")
                    log_file.flush()
                time.sleep(1)  # Log every second
        except Exception as e:
            print(f"GPU Monitor Stopped: {e}")
        finally:
            pynvml.nvmlShutdown()  # Shutdown NVML


def build_or_load_tokenizer(config, dataset, language):
    """Build or load a tokenizer for the given language"""
    tokenizer_path = os.path.join(config["tokenizer_dir"], f"tokenizer_{language}.json")
    
    if not os.path.exists(tokenizer_path):
        # Create directory if it doesn't exist
        os.makedirs(config["tokenizer_dir"], exist_ok=True)
        
        # Initialize tokenizer
        tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.WordLevelTrainer(
            special_tokens=["<unk>", "<pad>", "<sos>", "<eos>"], 
            min_frequency=2
        )
        
        # Get sentences for training
        sentences = [item["translation"][language] for item in dataset]
        tokenizer.train_from_iterator(sentences, trainer=trainer)
        tokenizer.save(tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    
    return tokenizer


def compute_bleu(references, predictions, tokenizer_tgt=None):
    """Compute BLEU score for model evaluation"""
    metric = evaluate.load("sacrebleu")
    
    # If predictions are token IDs, decode them
    if isinstance(predictions[0], torch.Tensor):
        decoded_preds = [tokenizer_tgt.decode(pred.tolist()) for pred in predictions]
    else:
        decoded_preds = predictions
    
    # Format for sacrebleu (expects list of references for each prediction)
    formatted_refs = [[ref] for ref in references]
    
    result = metric.compute(predictions=decoded_preds, references=formatted_refs)
    return result["score"]


def greedy_decode(model, src, src_mask, tokenizer_tgt, max_len, device):
    """
    Perform greedy decoding for inference
    """
    sos_tkn_id = tokenizer_tgt.token_to_id("<sos>")
    eos_tkn_id = tokenizer_tgt.token_to_id("<eos>")
    
    # Get encoder output
    encoder_output = model.encode(src, src_mask)
    
    # Start with SOS token
    decoder_input = torch.full((1, 1), sos_tkn_id, dtype=torch.long, device=device)
    
    while decoder_input.size(1) < max_len:
        # Create causal mask for decoder
        causal_mask = generate_causal_mask(decoder_input.size(1)).to(device)
        
        # Get decoder output
        decoder_output = model.decode(encoder_output, src_mask, decoder_input, causal_mask)
        
        # Get next token prediction
        next_token_logits = model.linear_proj(decoder_output[:, -1])
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
        
        # Append to decoder input
        decoder_input = torch.cat([decoder_input, next_token], dim=1)
        
        # Stop if EOS token is generated
        if next_token.item() == eos_tkn_id:
            break
    
    return decoder_input.squeeze(0)


def validate(model, val_loader, tokenizer_src, tokenizer_tgt, config, device):
    """
    Validate the model on the validation set
    """
    model.eval()
    val_loss = 0
    references = []
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move batch to device
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            causal_mask = batch["causal_mask"].to(device)
            target = batch["target"].to(device)
            
            # Forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, causal_mask)
            logits = model.linear_proj(decoder_output)
            
            # Calculate loss
            loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("<pad>"))
            loss = loss_fn(logits.view(-1, tokenizer_tgt.get_vocab_size()), target.view(-1))
            val_loss += loss.item()
            
            # Generate translations for BLEU calculation
            for i in range(encoder_input.size(0)):
                src = encoder_input[i:i+1]
                src_mask = encoder_mask[i:i+1]
                
                # Get reference text
                ref_text = batch["tgt_text"][i]
                references.append(ref_text)
                
                # Generate prediction
                pred_tokens = greedy_decode(model, src, src_mask, tokenizer_tgt, config["seq_len"], device)
                pred_text = tokenizer_tgt.decode(pred_tokens.detach().cpu().tolist())
                predictions.append(pred_text)
    
    # Calculate average loss and BLEU score
    avg_loss = val_loss / len(val_loader)
    bleu_score = compute_bleu(references, predictions)
    
    return avg_loss, bleu_score, references, predictions


def load_config(config_file):
    """
    Load configuration from YAML file
    """
    with open(config_file, 'r', encoding="utf-8") as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def train_model_with_dataset(config):
    """
    Train a translation model using a Hugging Face dataset
    """
    # Extract config values
    direction = config.get("direction", "en-es")
    source_lang = direction.split("-")[0]
    target_lang = direction.split("-")[1]
    
    dataset_name = config.get("dataset_name", "opus100")
    dataset_config = config.get("dataset_config", direction)
    
    output_dir = config.get("output_dir", f"custom-translation-{direction}")
    max_length = config.get("max_length", 128)
    
    gpu_log_file = config.get("gpu_log_file", "gpu_usage.log")
    
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Start GPU monitoring if available
    if torch.cuda.is_available() and pynvml is not None:
        gpu_thread = threading.Thread(target=gpu_monitor, args=(gpu_log_file,))
        gpu_thread.daemon = True
        gpu_thread.start()
        print(f"GPU monitoring enabled, logging to {gpu_log_file}")
    
    # Load dataset
    print(f"Loading dataset {dataset_name}/{dataset_config}")
    raw_dataset = load_dataset(dataset_name, dataset_config)
    
    # Prepare tokenizers
    config["tokenizer_dir"] = os.path.join(output_dir, "tokenizers")
    tokenizer_src = build_or_load_tokenizer(config, raw_dataset["train"], source_lang)
    tokenizer_tgt = build_or_load_tokenizer(config, raw_dataset["train"], target_lang)
    
    print(f"Source vocabulary size: {tokenizer_src.get_vocab_size()}")
    print(f"Target vocabulary size: {tokenizer_tgt.get_vocab_size()}")
    
    # Create train and validation datasets
    train_dataset = LanguagePairDataset(
        raw_dataset["train"], 
        tokenizer_src, 
        tokenizer_tgt, 
        source_lang, 
        target_lang, 
        config.get("seq_len", 350)
    )
    
    val_dataset = LanguagePairDataset(
        raw_dataset["validation"], 
        tokenizer_src, 
        tokenizer_tgt, 
        source_lang, 
        target_lang, 
        config.get("seq_len", 350)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.get("batch_size", 16), 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.get("val_batch_size", 8), 
        shuffle=False
    )
    
    # Initialize model
    model = build(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        config.get("seq_len", 350),
        config.get("seq_len", 350),
        config.get("hidden_size", 512),
        config.get("attention_dropout", 0.1),
        config.get("num_attention_heads", 8),
        config.get("num_hidden_layers", 6),
        config.get("intermediate_size", 2048)
    ).to(device)
    
    # Initialize optimizer
    optimizer = Adam(
        model.parameters(), 
        lr=config.get("learning_rate", 5e-5),
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=2, 
        verbose=True
    )
    
    # Initialize loss function
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id("<pad>"),
        label_smoothing=0.1
    ).to(device)
    
    # Initialize early stopping
    early_stopping = EarlyStoppingCallback(patience=config.get("patience", 3))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Track best model
    best_bleu = 0.0
    
    # Training loop
    print("Starting training...")
    for epoch in range(config.get("num_epochs", 5)):
        model.train()
        total_loss = 0
        
        # Training step
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.get('num_epochs', 5)}")
        for batch in progress_bar:
            # Move batch to device
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            causal_mask = batch["causal_mask"].to(device)
            target = batch["target"].to(device)
            
            # Forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, causal_mask)
            logits = model.linear_proj(decoder_output)
            
            # Calculate loss
            loss = loss_fn(logits.view(-1, tokenizer_tgt.get_vocab_size()), target.view(-1))
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate average training loss
        avg_train_loss = total_loss / len(train_loader)
        print(f"Training loss: {avg_train_loss:.4f}")
        
        # Validation step
        val_loss, bleu_score, references, predictions = validate(
            model, val_loader, tokenizer_src, tokenizer_tgt, config, device
        )
        
        print(f"Validation loss: {val_loss:.4f}, BLEU score: {bleu_score:.2f}")
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Save model if it's the best so far
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            model_path = os.path.join(output_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'bleu_score': bleu_score,
            }, model_path)
            print(f"Best model saved with BLEU score: {bleu_score:.2f}")
            
            # Save some example translations
            with open(os.path.join(output_dir, "examples.txt"), "w", encoding="utf-8") as f:
                for i in range(min(5, len(references))):
                    f.write(f"Reference: {references[i]}\n")
                    f.write(f"Prediction: {predictions[i]}\n")
                    f.write("---\n")
        
        # Check for early stopping
        if early_stopping.check(val_loss):
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint for this epoch
        checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'bleu_score': bleu_score,
        }, checkpoint_path)
    
    print(f"Training completed. Best BLEU score: {best_bleu:.2f}")
    
    # Load best model for testing
    best_model_path = os.path.join(output_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model with BLEU score: {checkpoint['bleu_score']:.2f}")
    
    # Test translations with sample sentences
    test_translations(model, tokenizer_src, tokenizer_tgt, source_lang, target_lang, device)


def test_translations(model, tokenizer_src, tokenizer_tgt, source_lang, target_lang, device):
    """Test the model with example sentences"""
    model.eval()
    
    # Example sentences
    if source_lang == "en":
        test_sentences = [
            "Hello, how are you today?",
            "Machine translation is an interesting field of natural language processing.",
            "I would like to visit Spain next summer."
        ]
    else:
        test_sentences = [
            "Hola, ¿cómo estás hoy?",
            "La traducción automática es un campo interesante del procesamiento del lenguaje natural.",
            "Me gustaría visitar España el próximo verano."
        ]
    
    print("\nTest translations:")
    print("-" * 50)
    
    with torch.no_grad():
        for sentence in test_sentences:
            # Tokenize input
            encoder_input = tokenizer_src.encode(sentence).ids
            encoder_input = torch.tensor([tokenizer_src.token_to_id("<sos>")] + 
                                         encoder_input + 
                                         [tokenizer_src.token_to_id("<eos>")], 
                                         device=device).unsqueeze(0)
            
            # Create source mask
            encoder_mask = (encoder_input != tokenizer_src.token_to_id("<pad>")).unsqueeze(1).unsqueeze(1).int().to(device)
            
            # Generate translation
            translation_tokens = greedy_decode(model, encoder_input, encoder_mask, 
                                              tokenizer_tgt, max_len=100, device=device)
            
            # Decode translation
            translation = tokenizer_tgt.decode(translation_tokens.tolist())
            translation = translation.replace("<sos>", "").replace("<eos>", "").strip()
            
            print(f"Source: {sentence}")
            print(f"Translation: {translation}")
            print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune custom Transformer translation model")
    parser.add_argument("--config", default="config.yaml", help="Configuration file (YAML)")
    parser.add_argument("--direction", help="Translation direction (e.g., 'en-es')")
    parser.add_argument("--dataset", help="Dataset to use (e.g., 'opus100')")
    parser.add_argument("--output_dir", help="Directory to save model")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments if provided
    if args.direction:
        config["direction"] = args.direction
    if args.dataset:
        config["dataset_name"] = args.dataset
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    # Train and test the model
    train_model_with_dataset(config)


if __name__ == "__main__":
    main()