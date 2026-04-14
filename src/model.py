"""
model.py
========
HuggingFace transformers 기반 ViT-num-spec 시계열 예측 모델.

사전학습 모델: google/vit-base-patch16-224 (HuggingFace Hub 자동 다운로드)

구조:
  ViTModel (backbone, 사전학습)
    └─ CLS 토큰 [B, 768]
         └─ MLP head [B, forecast_len]  ← 랜덤 초기화, MSE 학습
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class VisionTransformerForecaster(nn.Module):
    """
    ViT-num-spec: 스펙트로그램 이미지 → 다단계 시계열 예측.

    Parameters
    ----------
    hf_model_name : str
        HuggingFace 모델 ID 또는 로컬 디렉토리 경로.
        기본: "google/vit-base-patch16-224"
    forecast_len  : int
        예측 horizon.  Synthetic/Financial=20, Temperature=10
    freeze_backbone : bool
        True 이면 ViT backbone 파라미터를 동결 (head 만 학습).
        False 이면 전체 fine-tuning.
    """

    def __init__(
        self,
        hf_model_name: str = "google/vit-base-patch16-224",
        forecast_len:  int  = 20,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.forecast_len = forecast_len

        # ── Backbone: HuggingFace ViT ────────────────────────────────────────
        # .bin 파일 경로면 111.py 방식으로 로컬 로드, 아니면 Hub/디렉토리 로드
        if hf_model_name.endswith(".bin"):
            config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
            self.vit = ViTModel(config)
            state_dict = torch.load(hf_model_name, map_location="cpu")
            self.vit.load_state_dict(state_dict)
        else:
            self.vit = ViTModel.from_pretrained(hf_model_name)
        hidden_size = self.vit.config.hidden_size   # ViT-B = 768

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        # ── 회귀 헤드: CLS 토큰 임베딩 → forecast_len 스칼라 ────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, forecast_len),
        )
        self._init_head()

    # ── 초기화 ────────────────────────────────────────────────────────────────
    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(
        self,
        x:       torch.Tensor,
        targets: torch.Tensor | None = None,
    ):
        """
        Parameters
        ----------
        x       : FloatTensor [B, 3, H, W]  스펙트로그램 이미지 (128×128)
        targets : FloatTensor [B, forecast_len]  학습 시 제공

        Returns
        -------
        학습 시 : loss scalar (MSE)
        추론 시 : preds [B, forecast_len]
        """
        # interpolate_pos_encoding=True → 128×128 입력을 224 학습 pos_embed 에 보간
        outputs = self.vit(
            pixel_values=x,
            interpolate_pos_encoding=True,
        )
        cls_out = outputs.last_hidden_state[:, 0]   # [B, 768]
        preds   = self.head(cls_out)                 # [B, forecast_len]

        if targets is not None:
            return nn.functional.mse_loss(preds, targets)
        return preds

    # ── 파라미터 수 ───────────────────────────────────────────────────────────
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if (p.requires_grad or not trainable_only)
        )


# ── 빠른 확인 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = VisionTransformerForecaster(
        hf_model_name="google/vit-base-patch16-224",
        forecast_len=20,
    )
    print(f"학습 파라미터 수: {model.num_parameters() / 1e6:.1f}M")

    dummy = torch.randn(4, 3, 128, 128)
    preds = model(dummy)
    print(f"추론 출력 shape : {preds.shape}")   # (4, 20)

    tgt  = torch.randn(4, 20)
    loss = model(dummy, tgt)
    print(f"학습 loss       : {loss.item():.4f}")
