# Download from: https://huggingface.co/datasets/doof-ferb/fpt_fosd

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
import torchaudio.transforms as T
from datasets import load_dataset
from torch.utils.data import Dataset

from src.utils import audio_utils


@dataclass
class FPTFOSDBatch:
    """Batch container for self-supervised FPTFOSD learning with masking."""

    waveforms: torch.Tensor
    spectrograms: torch.Tensor
    context_masks: torch.Tensor
    prediction_masks: List[Tuple[torch.Tensor]]
    targets: torch.Tensor
    audio_names: List[str]

    # add a method to display the batch
    def __str__(self):
        str = f"FPTFOSD batch:\n"
        str += f" - Batch size:                                   {self.waveforms.shape[0]}\n"
        str += (
            f" - Waveforms [B, samples]:                       {self.waveforms.shape}\n"
        )
        str += f" - Spectrograms [B, C, T, n_mels]:               {self.spectrograms.shape}\n"
        str += f" - Context masks n_masks * [B, n_patches]:       {len(self.context_masks)} * {self.context_masks[0].shape}\n"
        str += f" - Prediction masks n_masks * [B, n_patches]:    {len(self.prediction_masks)} * {self.prediction_masks[0].shape}\n"
        # str += f" - Context masks: [B, n_patches]:      {self.context_masks.shape}\n"
        # str += f" - Target masks: [B, ?]:               {self.prediction_masks.shape}\n"
        # str += f" - N prediction masks:                 {len(self.prediction_masks[0])}\n"
        str += (
            f" - Targets [B, n_classes]:                       {self.targets.shape}\n"
        )
        str += f" - Audio names:                                  {len(self.audio_names)} names\n"
        return str


class FPTFOSDDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        sr: int = 32000,
        clip_length: int = 10,
        transforms: Optional[List[torch.nn.Module]] = None,
    ):
        super().__init__()
        self.dataset_id = "doof-ferb/fpt_fosd"
        self.sr = sr
        self.clip_length = clip_length * sr
        self.transforms = transforms
        self.dataset
        self.length = len(self.dataset)

    def _load_sample(self, data_dir: Path | None) -> Dataset:
        if data_dir is not None:
            dataset = load_dataset("parquet", data_dir=data_dir)["train"]
        else:
            dataset = load_dataset(self.dataset_id)["train"]
        return dataset

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict:

        sample = self.dataset[idx]
        audio_data = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]

        transcription = sample["transcription"]

        waveform = torch.from_numpy(audio_data).float()
        waveform = audio_utils.normalize_audio(waveform)
        waveform = self.resample(waveform) if self.sr != sampling_rate else waveform

        item = {"waveform": waveform, "transcription": transcription}

        # Apply transforms if specified
        if self.transforms:
            item["transformed_waveform"] = self._apply_transforms(waveform)

        return item

    def _load_waveform(self, idx: int) -> torch.Tensor:
        """Load and preprocess audio waveform"""
        if self.mp3_dataset:
            arr = audio_utils.decode_mp3(self.dataset_file["mp3"][idx])
        else:
            arr = audio_utils.int16_to_float32(self.dataset_file["waveform"][idx])

        waveform = torch.from_numpy(arr).float()
        waveform = audio_utils.normalize_audio(waveform)
        waveform = self.resample(waveform) if self.sr != 32000 else waveform
        return audio_utils.pad_or_truncate(waveform, self.clip_length)

    def _apply_transforms(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply transformation pipeline to waveform"""
        # Add batch dimension for transform compatibility
        transformed = waveform.clone().unsqueeze(0)

        for transform in self.transforms:
            transformed = transform(transformed)

        return transformed

    def __del__(self):
        if self.dataset_file is not None:
            self.dataset_file.close()


def collate_FPTFOSD_batch(batch: List[Union[Dict, Tuple]]) -> FPTFOSDBatch:
    """Collate function for FPTFOSD batches.

    Handles both dictionary and tuple batch formats
    """
    batch = torch.utils.data.default_collate(batch)

    if "transformed_waveform" in batch:
        return FPTFOSDBatch(
            waveforms=batch["waveform"],
            spectrograms=batch["transformed_waveform"],
            context_masks=None,
            prediction_masks=None,
            targets=batch["target"],
            audio_names=batch["audio_name"],
        )

    else:
        return FPTFOSDBatch(
            waveforms=batch["waveform"],
            spectrograms=None,
            context_masks=None,
            prediction_masks=None,
            targets=batch["target"],
            audio_names=batch["audio_name"],
        )
