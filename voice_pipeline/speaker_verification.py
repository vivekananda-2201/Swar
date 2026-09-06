"""
Speaker Recognition & Voiceprint Verification for Swar Voice Pipeline.

Uses CAM++ (D-TDNN) ONNX model (~28MB, ~3ms CPU latency) to extract 192-dimensional
speaker embeddings. Supports:
1. Multi-user authorized voiceprint enrollment and whitelist verification.
2. Anti-Kokoro assistant echo rejection (negative match filter) for open laptop speakers.
3. Cosine similarity matching between live microphone audio and registered voiceprints.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch
import torchaudio.compliance.kaldi as kaldi
import torchaudio.functional as F

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path.home() / ".cache" / "swar" / "models"
DEFAULT_MODEL_FILE = "campplus_cn_en_common_200k.onnx"
DEFAULT_VOICEPRINTS_DIR = Path.home() / ".cache" / "swar" / "voiceprints"
HF_MODEL_REPO = "welcomyou/campplus-3dspeaker-200k-onnx"


class SpeakerEmbeddingExtractor:
    """
    Extracts L2-normalized 192-dimensional speaker embeddings using CAM++ ONNX.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = self._resolve_model_path(model_path)
        logger.info(f"Loading Speaker Embedding model from {self.model_path}...")

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 2
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _resolve_model_path(self, model_path: Optional[Union[str, Path]]) -> Path:
        if model_path:
            p = Path(os.path.expanduser(str(model_path)))
            if p.is_file():
                return p

        # Check default cache location
        default_path = DEFAULT_MODEL_DIR / DEFAULT_MODEL_FILE
        if default_path.is_file():
            return default_path

        # Download from HuggingFace Hub
        logger.info("Speaker verification model not found locally; downloading from Hugging Face...")
        DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=DEFAULT_MODEL_FILE,
            local_dir=str(DEFAULT_MODEL_DIR),
        )
        return Path(downloaded)

    def extract(self, audio: Union[np.ndarray, torch.Tensor], sample_rate: int = 16000) -> np.ndarray:
        """
        Extracts 192-dim L2-normalized speaker embedding from raw audio.
        Audio can be int16 or float32. Resamples to 16kHz if necessary.
        """
        if isinstance(audio, np.ndarray):
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            wav_tensor = torch.from_numpy(audio).float()
        else:
            wav_tensor = audio.float()

        if wav_tensor.ndim == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        elif wav_tensor.ndim == 2 and wav_tensor.shape[0] > wav_tensor.shape[1]:
            wav_tensor = wav_tensor.t()

        # Resample to 16000 Hz if needed
        if sample_rate != 16000:
            wav_tensor = F.resample(wav_tensor, sample_rate, 16000)

        # Ensure minimum length of ~0.4s (6400 samples at 16kHz) for stable fbank extraction
        min_samples = 6400
        if wav_tensor.shape[1] < min_samples:
            pad_amount = min_samples - wav_tensor.shape[1]
            wav_tensor = torch.nn.functional.pad(wav_tensor, (0, pad_amount), mode="constant", value=0.0)

        # Scale float32 [-1, 1] to 16-bit range for Kaldi fbank
        scaled_wav = wav_tensor * (1 << 15)

        # Compute 80-dimensional log Mel filterbank
        fbank = kaldi.fbank(scaled_wav, num_mel_bins=80, sample_frequency=16000)
        # CMVN: Zero-mean normalization over time frames
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        fbank_np = fbank.unsqueeze(0).numpy().astype(np.float32)

        # ONNX inference
        raw_embs = self.session.run([self.output_name], {self.input_name: fbank_np})[0]
        emb = raw_embs[0].flatten()

        # L2-normalize
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm
        return emb


class VoiceprintManager:
    """
    Manages user voiceprints and assistant (Kokoro) reference voiceprints.
    Provides fast cosine similarity checks for multi-user whitelist and anti-echo blacklist.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        voiceprints_dir: Optional[Union[str, Path]] = None,
    ):
        self.extractor = SpeakerEmbeddingExtractor(model_path)
        self.voiceprints_dir = Path(os.path.expanduser(str(voiceprints_dir or DEFAULT_VOICEPRINTS_DIR)))
        self.voiceprints_dir.mkdir(parents=True, exist_ok=True)

        self.user_voiceprints: Dict[str, np.ndarray] = {}
        self.kokoro_voiceprints: Dict[str, np.ndarray] = {}
        self.kokoro_voiceprint: Optional[np.ndarray] = None
        self.current_kokoro_voice: Optional[str] = None

        self.load_all_user_voiceprints()
        self.load_all_kokoro_voiceprints()

    def load_all_user_voiceprints(self) -> int:
        """Loads all saved user .npy voiceprint files from voiceprints_dir."""
        self.user_voiceprints.clear()
        count = 0
        if self.voiceprints_dir.is_dir():
            for p in self.voiceprints_dir.glob("*.npy"):
                try:
                    emb = np.load(p)
                    norm = np.linalg.norm(emb)
                    if norm > 1e-6:
                        emb = emb / norm
                    name = p.stem
                    self.user_voiceprints[name] = emb
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to load user voiceprint {p}: {e}")
        logger.debug(f"Loaded {count} user voiceprint(s) from {self.voiceprints_dir}.")
        return count

    def load_all_kokoro_voiceprints(self) -> int:
        """Loads all cached Kokoro reference and acoustic voiceprints from DEFAULT_MODEL_DIR."""
        self.kokoro_voiceprints.clear()
        count = 0
        if DEFAULT_MODEL_DIR.is_dir():
            for p in DEFAULT_MODEL_DIR.glob("kokoro_*.npy"):
                try:
                    emb = np.load(p)
                    norm = np.linalg.norm(emb)
                    if norm > 1e-6:
                        emb = emb / norm
                    name = p.stem.replace("kokoro_", "")
                    self.kokoro_voiceprints[name] = emb
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to load Kokoro voiceprint {p}: {e}")

        if self.kokoro_voiceprints:
            self.kokoro_voiceprint = next(iter(self.kokoro_voiceprints.values()))
        else:
            self.kokoro_voiceprint = None
        logger.debug(f"Loaded {count} Kokoro voiceprint(s) from {DEFAULT_MODEL_DIR}.")
        return count

    def enroll_user(self, name: str, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Enrolls a new user voiceprint from audio samples and saves it to disk.
        """
        clean_name = name.strip().lower().replace(" ", "_")
        emb = self.extractor.extract(audio, sample_rate=sample_rate)
        self.user_voiceprints[clean_name] = emb

        save_path = self.voiceprints_dir / f"{clean_name}.npy"
        np.save(save_path, emb)
        logger.info(f"User voiceprint '{clean_name}' saved to {save_path}.")
        return emb

    def delete_user(self, name: str) -> bool:
        """Removes an enrolled user voiceprint."""
        clean_name = name.strip().lower().replace(" ", "_")
        deleted = False
        if clean_name in self.user_voiceprints:
            del self.user_voiceprints[clean_name]
            deleted = True
        save_path = self.voiceprints_dir / f"{clean_name}.npy"
        if save_path.is_file():
            save_path.unlink()
            deleted = True
        return deleted

    def delete_kokoro_voiceprint(self, voice_name: str) -> bool:
        """Removes an enrolled Kokoro voiceprint (both acoustic and digital files)."""
        clean_name = voice_name.strip().lower().replace(" ", "_").replace("kokoro_", "")
        deleted = False

        keys_to_delete = [k for k in self.kokoro_voiceprints if k == clean_name or k == f"{clean_name}_digital"]
        for k in keys_to_delete:
            del self.kokoro_voiceprints[k]
            deleted = True

        acoustic_path = DEFAULT_MODEL_DIR / f"kokoro_{clean_name}.npy"
        digital_path = DEFAULT_MODEL_DIR / f"kokoro_{clean_name}_digital.npy"
        if acoustic_path.is_file():
            acoustic_path.unlink()
            deleted = True
        if digital_path.is_file():
            digital_path.unlink()
            deleted = True

        if self.kokoro_voiceprints:
            self.kokoro_voiceprint = next(iter(self.kokoro_voiceprints.values()))
        else:
            self.kokoro_voiceprint = None
        return deleted

    def list_users(self) -> List[str]:
        """Returns list of enrolled speaker names."""
        return list(self.user_voiceprints.keys())

    def list_kokoro_voices(self) -> List[str]:
        """Returns list of enrolled Kokoro voice names."""
        voices = set()
        for k in self.kokoro_voiceprints.keys():
            base = k.replace("_digital", "")
            voices.add(base)
        return sorted(voices)

    def matches_authorized_user(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        threshold: float = 0.65,
        authorized_names: Optional[List[str]] = None,
    ) -> Tuple[bool, Optional[str], float]:
        """
        Checks if audio matches any enrolled authorized user.
        
        Returns:
            (is_match: bool, matched_name: Optional[str], similarity_score: float)
        """
        if not self.user_voiceprints:
            # Fallback if no voiceprints have been enrolled yet
            return True, None, 1.0

        emb = self.extractor.extract(audio, sample_rate=sample_rate)

        best_name: Optional[str] = None
        best_score = -1.0

        for name, user_emb in self.user_voiceprints.items():
            if authorized_names and name not in authorized_names:
                continue
            sim = float(np.dot(emb, user_emb))
            if sim > best_score:
                best_score = sim
                best_name = name

        is_match = (best_score >= threshold)
        return is_match, best_name, max(0.0, best_score)

    def ensure_kokoro_voiceprint(self, kokoro_engine, voice_name: str) -> np.ndarray:
        """
        Ensures Kokoro TTS reference voiceprint is loaded and cached for the active voice.
        Synthesizes a short reference utterance if not already cached.
        """
        clean_name = voice_name.lower().strip()
        cache_file = DEFAULT_MODEL_DIR / f"kokoro_{clean_name}.npy"

        if clean_name in self.kokoro_voiceprints:
            self.kokoro_voiceprint = self.kokoro_voiceprints[clean_name]
            self.current_kokoro_voice = clean_name
            return self.kokoro_voiceprint

        if cache_file.is_file():
            try:
                emb = np.load(cache_file)
                norm = np.linalg.norm(emb)
                if norm > 1e-6:
                    emb = emb / norm
                self.kokoro_voiceprints[clean_name] = emb
                self.kokoro_voiceprint = emb
                self.current_kokoro_voice = clean_name
                return emb
            except Exception as e:
                logger.debug(f"Could not load cached Kokoro voiceprint ({e}); re-synthesizing...")

        logger.info(f"Synthesizing Kokoro reference voiceprint for voice '{clean_name}'...")
        ref_text = (
            "Hello, I am your voice assistant powered by Swar. "
            "This reference sample allows acoustic echo rejection during speaker playback."
        )
        audio = kokoro_engine.synthesize_chunk(ref_text, voice=clean_name, speed=1.0)
        if audio is None:
            raise RuntimeError(f"Failed to synthesize reference audio for Kokoro voice '{clean_name}'")

        # Kokoro produces 24kHz audio; extract embedding will resample to 16kHz
        emb = self.extractor.extract(audio, sample_rate=24000)
        self.kokoro_voiceprints[clean_name] = emb
        self.kokoro_voiceprint = emb
        self.current_kokoro_voice = clean_name

        try:
            DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, emb)
            logger.info(f"Kokoro reference voiceprint saved to {cache_file}.")
        except Exception as e:
            logger.warning(f"Could not persist Kokoro voiceprint to cache: {e}")

        return emb

    def is_kokoro_echo(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        threshold: float = 0.60,
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Checks if audio matches any enrolled/cached Kokoro TTS voiceprint.
        
        Returns:
            (is_kokoro: bool, similarity_score: float, matched_voice: Optional[str])
        """
        if not self.kokoro_voiceprints and self.kokoro_voiceprint is None:
            return False, 0.0, None

        emb = self.extractor.extract(audio, sample_rate=sample_rate)

        best_score = -1.0
        best_voice = None

        targets = self.kokoro_voiceprints if self.kokoro_voiceprints else {"default": self.kokoro_voiceprint}

        for vname, v_emb in targets.items():
            sim = float(np.dot(emb, v_emb))
            if sim > best_score:
                best_score = sim
                best_voice = vname

        is_kokoro = (best_score >= threshold)
        matched = best_voice if is_kokoro else None
        return is_kokoro, max(0.0, best_score), matched
