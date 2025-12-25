import os
import logging
from typing import Dict, Tuple, List, Any
import io
import gc

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import ray
import boto3

from sasrec_data import load_sessions_for_training, build_vocab, SASRecDataset, MAX_LEN
from sasrec_model import SASRec

logging.basicConfig(level=logging.INFO)

s3_uri = os.environ["S3_URI"]
if not s3_uri.endswith("/"):
    s3_uri += "/"

SASREC_PREFIX = os.getenv("SASREC_S3_PREFIX")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


def _parse_s3_uri(uri: str):
    # "s3://bucket/prefix/file" -> ("bucket", "prefix/file")
    assert uri.startswith("s3://")
    bucket_key = uri[5:]
    bucket, key = bucket_key.split("/", 1)
    return bucket, key


# Initialize Ray once in the driver
ray.init(address=os.getenv("RAY_ADDRESS", "local"), ignore_reinit_error=True)


@ray.remote
def load_and_preprocess_data():
    """Load sessions and build vocabulary in a remote function to reduce memory pressure."""
    logging.info("Loading sessions from MongoDB...")
    sessions = load_sessions_for_training()
    if not sessions:
        logging.warning("No sessions to train on.")
        return None, None, None
    
    logging.info(f"Loaded {len(sessions)} sessions")
    logging.info("Building vocabulary...")
    item2id, id2item = build_vocab(sessions)
    logging.info(f"Vocabulary size: {len(item2id)} items")
    
    return sessions, item2id, id2item


@ray.remote
def train_sasrec_model(sessions: List[Dict[str, Any]], item2id: Dict[str, int], 
                       id2item: Dict[int, str], epochs: int = 3, 
                       batch_size: int = 128, lr: float = 1e-3):
    """Train SASRec model in a remote function."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Create dataset
    dataset = SASRecDataset(sessions, item2id, max_len=MAX_LEN)
    if len(dataset) == 0:
        logging.warning("No valid sequences after preprocessing.")
        return None
    
    logging.info(f"Dataset size: {len(dataset)} sequences")
    
    # Create data loader with smaller batch size to reduce memory
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    num_items = len(item2id)
    
    # Create model
    model = SASRec(num_items=num_items, max_len=MAX_LEN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for input_seq, target_seq in loader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            logits = model.predict_next(input_seq)  # (B, num_items+1)
            B, L = target_seq.size()
            target_next = target_seq[:, -1]  # next item at last position

            # ignore positions where target is padding (0)
            valid_mask = target_next > 0
            if valid_mask.sum() == 0:
                continue

            logits_valid = logits[valid_mask]
            targets_valid = target_next[valid_mask]

            loss = F.cross_entropy(logits_valid, targets_valid)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        logging.info(f"[SASRec] Epoch {epoch + 1}/{epochs}, loss={avg_loss:.4f}")
        
        # Clear cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Create checkpoint
    ckpt = {
        "model_state": model.state_dict(),
        "item2id": item2id,
        "id2item": id2item,
        "max_len": MAX_LEN,
    }
    
    # Clear model from memory
    del model, dataset, loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return ckpt


def upload_checkpoint(ckpt: Dict):
    """Upload checkpoint to S3."""
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    
    buf = io.BytesIO()
    torch.save(ckpt, buf)
    buf.seek(0)

    model_uri = f"{s3_uri.rstrip('/')}/{SASREC_PREFIX}/sasrec_latest.pt"
    bucket, key = _parse_s3_uri(model_uri)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    logging.info(f"SASRec model uploaded to {model_uri}")


if __name__ == "__main__":
    try:
        # Load and preprocess data in a remote function
        sessions, item2id, id2item = ray.get(load_and_preprocess_data.remote())
        
        if sessions is None:
            logging.warning("No data to train on. Exiting.")
            exit(0)
        
        # Train model in a remote function
        ckpt = ray.get(train_sasrec_model.remote(sessions, item2id, id2item))
        
        if ckpt is None:
            logging.error("Training failed - no checkpoint returned")
            exit(1)
        
        # Upload checkpoint (runs in main process)
        upload_checkpoint(ckpt)
        
        logging.info("SASRec training completed successfully!")
    except Exception as e:
        logging.error(f"SASRec training failed: {e}", exc_info=True)
        raise
    finally:
        # Clean up Ray resources
        logging.info("Shutting down Ray...")
        ray.shutdown()
        logging.info("Ray cluster shut down. Exiting container.")
