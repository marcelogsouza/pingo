#!/usr/bin/env bash
# Levanta a WebUI do Pingo (chat em / e o curso em /lab) no terminal: Ctrl+C desliga.
# Atalho para `uv run python scripts/webui.py` — com uma bondade: derruba a instância
# anterior nossa antes de subir (um servidor por porta), e avisa da regra da VRAM.
#   ./scripts/webui.sh                      # device auto: GPU se a VRAM estiver livre
#   ./scripts/webui.sh --device cpu         # com o professor/Strata servindo (regra do projeto)
#   ./scripts/webui.sh --host 127.0.0.1     # fecha a LAN
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

antigos=$(pgrep -f "scripts/webui.py" || true)
if [[ -n $antigos ]]; then
  echo "desligando a instância anterior da WebUI (pids: ${antigos//$'\n'/ })…"
  pkill -f "scripts/webui.py" || true
  sleep 1   # a porta 8010 precisa voltar antes do bind
fi

echo "dica: com o Strata/professor de pé, rode com --device cpu (a VRAM é disputada)."
echo      # o Ctrl+C chega ao processo em foreground; o Flask dev já sabe parar sozinho
uv run python scripts/webui.py "$@"
echo "WebUI parada."
