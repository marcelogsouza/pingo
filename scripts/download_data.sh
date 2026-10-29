#!/usr/bin/env bash
# Passo 2 — baixa e filtra os dados de pré-treino do Pingo (Linux/macOS).
#
# Fonte (histórico): dataset `oscar-corpus/community-oscar` no Hugging Face,
# dump `2024-10`, arquivos `data/2024-10/pt/pt_part_1..100.jsonl.zst` (~42,7 GB).
# O dataset é gated (gate auto): exige conta HF + aceitar os termos 1x na página
# do dataset. Use `2023-40` como reserva se faltarem tokens (veja --dumps).
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || { echo "instale o uv: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
uv sync

if ! uv run hf auth whoami >/dev/null 2>&1; then
  echo "faça login no HF (e aceite o gate do Community OSCAR na página do dataset):"
  echo "  uv run hf auth login"
  exit 1
fi

# download retomável (~42,7 GB; pode interromper no meio e rodar de novo)
uv run python scripts/download_data.py --dumps 2024-10 --max-gb 45 --workers 4

# content → filtros de qualidade → data/raw/*.parquet (~2% marcado val)
uv run python scripts/process_data.py --dumps 2024-10
