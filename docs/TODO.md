# Project Planning: Vietnamese Speech-to-Text (STT)

## Phase 1: Data Collection & Preprocessing
- [ ] **Find & Download Datasets**: Gather open-source Vietnamese audio datasets (e.g., Common Voice Vietnamese, VIVOS, FOSD).
- [ ] **Data Cleaning & Normalization**:
  - [ ] Text normalization (convert to lowercase, remove punctuations, expand abbreviations/numbers).
  - [ ] Audio normalization (resample to 16kHz, convert to mono channel).
- [ ] **Feature Extraction**:
  - [ ] Write scripts to extract audio features (Mel-spectrograms or MFCCs).
  - [ ] Implement text tokenization (character-level or subword/BPE tokenization for Vietnamese).

## Phase 2: PyTorch Dataset & DataLoader
- [ ] **Create PyTorch `Dataset`**: Implement a custom dataset class to load audio features and text tokens.
- [ ] **Implement `collate_fn`**: Write a custom collate function to handle variable-length sequences (padding audio frames and text tokens).
- [ ] **Setup `DataLoader`**: Configure data loaders for train, validation, and test splits.

## Phase 3: Model Architecture Design
- [ ] **Research Architectures**: Decide on the baseline model (e.g., DeepSpeech2, Conformer, or a simple CNN+RNN architecture).
- [ ] **Implement Model in PyTorch**: 
  - [ ] Build the acoustic model (encoder).
  - [ ] Build the decoder (if using seq2seq, otherwise use a linear layer for CTC).

## Phase 4: Training Pipeline
- [ ] **Setup Loss Function**: Implement CTC Loss (Connectionist Temporal Classification) or Cross-Entropy Loss depending on the architecture.
- [ ] **Configure Optimizer & Scheduler**: Setup AdamW optimizer and a learning rate scheduler (e.g., OneCycleLR or ReduceLROnPlateau).
- [ ] **Write Training Loop**: Implement the main training loop with mixed precision (AMP) for faster training.
- [ ] **Write Validation Loop**: Track validation loss and implement early stopping.
- [ ] **Metrics Logging**: Integrate TensorBoard or Weights & Biases to track training progress.

## Phase 5: Evaluation & Decoding
- [ ] **Implement Metrics**: Add Word Error Rate (WER) and Character Error Rate (CER) calculation.
- [ ] **Decoding Strategies**:
  - [ ] Implement Greedy Decoding.
  - [ ] Implement Beam Search Decoding (optional: integrate a Vietnamese N-gram Language Model like KenLM to improve results).
- [ ] **Testing Script**: Write a script to evaluate the trained model on the test dataset.

## Phase 6: Inference & Demo
- [ ] **Inference Script**: Create a script that takes a raw `.wav` file as input and outputs the predicted text.
- [ ] **Live Demo (Optional)**: Build a simple UI using Gradio or Streamlit to test the STT model with a microphone.