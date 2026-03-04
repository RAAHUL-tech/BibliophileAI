# SASRec (Session-Based) Training

Trains a transformer-based sequence model (SASRec) on user sessions (sequences of book interactions) from MongoDB. The model is used at inference time to score the next items given the current session and produce session-based recommendations.

## Algorithm

- **SASRec**: Self-Attentive Sequential Recommendation. Each session is a sequence of item IDs (books). The model uses item embeddings + positional embeddings and a transformer encoder to produce a representation of the sequence; it scores the next item (e.g. via inner product with item embeddings). Training is typically next-item prediction (cross-entropy or BPR).
- **Sessions**: Built from `click_stream.events`: ordered by time per user/session, filtered by event types (read, page_turn, review, bookmark_add); each sequence is truncated/padded to a max length (e.g. 100).
- **Vocabulary**: Item IDs are mapped to integer indices; padding index 0; unknown/new items can be mapped to a special id.

## Implementation in this project

- **sasrec_data.py**: Loads events from MongoDB, groups by session (or user + time window), builds sequences; `build_vocab()` produces item2id/id2item; `SASRecDataset` yields (input_seq, target) for training.
- **sasrec_model.py**: PyTorch `nn.Module`: embedding layer, positional embedding, transformer encoder, output layer; forward returns sequence representation or next-item logits.
- **sasrec_train.py**:
  1. **Load**: Ray task loads sessions via `load_sessions_for_training()`, builds vocab.
  2. **Dataset**: `SASRecDataset(sessions, item2id, max_len=MAX_LEN)`.
  3. **Train**: DataLoader, optimizer (e.g. Adam), next-item loss (e.g. cross-entropy); trains for several epochs.
  4. **Upload**: Saves checkpoint (state_dict or full model) to S3 under `SASREC_S3_PREFIX`.
- **entrypoint.sh**: Starts Ray head, runs `python sasrec_train.py`, stops Ray.
- **Recommendation service**: `sasrec_inference.py` loads the checkpoint from S3, builds the same `SASRec` model, fetches the user’s current session from MongoDB, and returns top-k scored books.

## Environment

`MONGO_URI`, `S3_URI`, `SASREC_S3_PREFIX`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `RAY_ADDRESS`. Optional: epochs, batch_size, lr, max_len.
