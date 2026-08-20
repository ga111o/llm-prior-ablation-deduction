# Reconstruction error

## 설정
- 모델: `Qwen/Qwen3.5-27B` (BF16, 텍스트 전용 `AutoModelForCausalLM`)
- SAE: `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50` residual TopK, `d_sae=81920`, K=50
- 위치: layer 32 (50%), layer 54 (85%), 또는 둘 다 (`--sae-at 50|85|both`)
- GPU: DGX Spark GB10 (sm_121, ARM64, ~100GB). x86 개발 환경에서는 27B를 돌리지 않음

## 양자화
- 없음. SAE가 학습된 BF16 residual을 그대로 측정

## 데이터
- `NeelNanda/pile-10k` 스트리밍
- Qwen3.5 chat template로 user 메시지 래핑 (`enable_thinking=False`) 후 1024 토큰 청크
- 2048 시퀀스

## 측정
- 해당 층 residual → TopK SAE encode/decode → `x` vs `x̂`
- 잔차를 다음 층에 넣지 않음 (delta LM loss 아님)
- 주 지표 FVU = `E[||x-x̂||²] / E[||x-x̄||²]`
- 보조: cosine, relative L2, L0 (Top-K라 L0 ≈ 50이어야 함)

## 결과
- Spark에서 재측정. 이전 Gemma 4-bit 수치와 직접 비교하지 않음
