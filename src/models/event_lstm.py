"""LSTM-based win probability model for highlight extraction.

Loads event_lstm_best.pt trained in Colab (colab_train_event.py).
Exposes predict_win_prob_sequence() for per-event prob_win tracking.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "event_lstm_best.pt"
)


class EventEmbedding(nn.Module):
    def __init__(self, vocab: dict, embed_dim: int = 8):
        super().__init__()
        vocab_sizes = [
            vocab["num_event_types"],
            3,  # killer_team: 0(pad)/1(blue)/2(red)
            3,  # victim_team
            vocab["num_monster_types"],
            vocab["num_dragon_subtypes"],
            vocab["num_building_types"],
            vocab["num_tower_types"],
            vocab["num_lane_types"],
            vocab["num_kill_types"],
            vocab["num_ward_types"],
        ]
        self.embeddings = nn.ModuleList([
            nn.Embedding(vs, embed_dim, padding_idx=0) for vs in vocab_sizes
        ])
        num_num = 8  # timestamp, pos_x, pos_y, bounty, assists, kill_streak, gold_diff, xp_diff
        self.out_dim = len(vocab_sizes) * embed_dim + num_num

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        embedded = [emb(cat[:, :, i]) for i, emb in enumerate(self.embeddings)]
        cat_vec = torch.cat(embedded, dim=-1)
        return torch.cat([cat_vec, num], dim=-1)


class WinProbLSTM(nn.Module):
    def __init__(self, vocab: dict, embed_dim: int = 8,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.event_embedding = EventEmbedding(vocab, embed_dim)
        self.lstm = nn.LSTM(
            self.event_embedding.out_dim, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward_sequence(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        """모든 타임스텝의 승률을 반환. shape: (T,)"""
        x = self.event_embedding(cat, num)  # (1, T, input_size)
        lstm_out, _ = self.lstm(x)          # (1, T, H)
        out = self.dropout(lstm_out)
        logits = self.fc(out).squeeze(-1)   # (1, T)
        return self.sigmoid(logits).squeeze(0)  # (T,)


# ─────────────────────────────────────────────────────────────
# 로드 & 추론 API
# ─────────────────────────────────────────────────────────────

_lstm_model: Optional[WinProbLSTM] = None
_device = torch.device("cpu")


def load_lstm_model(path: str = MODEL_PATH) -> Optional[WinProbLSTM]:
    global _lstm_model
    if _lstm_model is not None:
        return _lstm_model

    if not os.path.exists(path):
        print(f"[WARN] LSTM model not found at {path}")
        return None

    try:
        ckpt = torch.load(path, map_location=_device)
        vocab = ckpt["vocab"]
        model = WinProbLSTM(
            vocab=vocab,
            embed_dim=ckpt.get("embed_dim", 8),
            hidden_size=ckpt.get("hidden_size", 128),
            num_layers=ckpt.get("num_layers", 2),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _lstm_model = model
        print(f"[INFO] LSTM model loaded from {path}")
        return _lstm_model
    except Exception as e:
        print(f"[ERROR] Failed to load LSTM model: {e}")
        return None


def predict_win_prob_sequence(
    cat_seq: np.ndarray,
    num_seq: np.ndarray,
) -> np.ndarray:
    """
    이벤트 시퀀스 전체의 timestep별 승률을 반환.

    Args:
        cat_seq: (N, 10) int64  — categorical features
        num_seq: (N, 8)  float32 — numerical features

    Returns:
        prob_win: (N,) float32  — 각 이벤트 직후 blue팀 승률
    """
    model = load_lstm_model()
    if model is None or len(cat_seq) == 0:
        return np.full(len(cat_seq), 0.5, dtype=np.float32)

    cat_t = torch.tensor(cat_seq, dtype=torch.long).unsqueeze(0)    # (1, N, 10)
    num_t = torch.tensor(num_seq, dtype=torch.float32).unsqueeze(0) # (1, N, 8)

    with torch.no_grad():
        probs = model.forward_sequence(cat_t, num_t)  # (N,)

    return probs.numpy().astype(np.float32)
