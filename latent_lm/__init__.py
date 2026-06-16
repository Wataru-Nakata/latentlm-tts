"""LatentLM TTS — next-token-diffusion text-to-speech.

An autoregressive Transformer emits a mixed sequence of discrete text tokens
(softmax LM head) and continuous 64-d acoustic latents (next-token diffusion
head). Latents come from a frozen VibeVoice σ-VAE; text conditions speech.

Public API:
    LatentLM, LatentLMConfig            — the model
    TorchBackbone, TorchBackboneConfig  — plain-torch backbone (no container)
    sample_tts, SampleConfig            — autoregressive inference
    TextTokenizer, TextTokenizerConfig
    VibeVoiceTokenizer                  — acoustic encoder/decoder
"""

from .data.text_tokenizer import TextTokenizer, TextTokenizerConfig
from .inference import SampleConfig, sample_tts
from .models.latent_lm import LatentLM, LatentLMConfig
from .models.tokenizer import VibeVoiceTokenizer
from .models.torch_backbone import TorchBackbone, TorchBackboneConfig

__all__ = [
    "LatentLM",
    "LatentLMConfig",
    "TorchBackbone",
    "TorchBackboneConfig",
    "sample_tts",
    "SampleConfig",
    "TextTokenizer",
    "TextTokenizerConfig",
    "VibeVoiceTokenizer",
]

__version__ = "0.1.0"
