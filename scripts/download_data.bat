@echo off
rem Passo 2 - baixa e filtra os dados de pre-treino do Pingo (Windows).
rem
rem Fonte (historico): dataset `oscar-corpus/community-oscar` no Hugging Face,
rem dump `2024-10`, arquivos `data/2024-10/pt/pt_part_1..100.jsonl.zst` (~42,7 GB).
rem O dataset e gated (gate auto): exige conta HF + aceitar os termos 1x na pagina
rem do dataset. Use `2023-40` como reserva se faltarem tokens (veja --dumps).
cd /d "%~dp0.."

where uv >nul 2>nul || (echo instale o uv: https://astral.sh/uv & exit /b 1)
uv sync || exit /b 1

uv run hf auth whoami >nul 2>nul
if errorlevel 1 (
  echo faca login no HF e aceite o gate do Community OSCAR na pagina do dataset:
  echo   uv run hf auth login
  exit /b 1
)

rem download retomavel (~42,7 GB; pode interromper no meio e rodar de novo)
uv run python scripts\download_data.py --dumps 2024-10 --max-gb 45 --workers 4 || exit /b 1

rem content - filtros de qualidade - data\raw\*.parquet (~2%% marcado val)
uv run python scripts\process_data.py --dumps 2024-10 || exit /b 1
