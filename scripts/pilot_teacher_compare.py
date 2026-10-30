"""Piloto: compara professores remotos (OpenRouter) no MESMO conjunto de trechos.

Não toca no run principal (`data/sft/sft-v1.jsonl`) nem no llama.cpp local: os
trechos são amostrados do corpus com seed própria, e cada modelo vê exatamente
os mesmos chunks/tipos/modos.

Uso:
    export OPENROUTER_API_KEY=...        # ou --key-file ~/.openrouter_key
    uv run python scripts/pilot_teacher_compare.py --n 10 \
        --models qwen/qwen3.7-flash deepseek/deepseek-v4.1-flash google/gemma-4-26b-a4b-it

Saída: `data/sft/or-pilot/<slug>.jsonl` (uma linha por CHAMADA, aceita ou não)
+ `<slug>.stats.json` + resumo no stdout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# gen_sft.py é script (não pacote): precisa entrar em sys.modules por causa dos @dataclass
_SPEC = importlib.util.spec_from_file_location(
    "gen_sft", Path(__file__).resolve().parent / "gen_sft.py")
g = importlib.util.module_from_spec(_SPEC)
sys.modules["gen_sft"] = g
_SPEC.loader.exec_module(g)

OR_URL = "https://openrouter.ai/api/v1/chat/completions"


def slug(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def load_key(args) -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key and args.key_file:
        key = Path(args.key_file).expanduser().read_text().strip()
    if not key:
        sys.exit("sem chave: exporte OPENROUTER_API_KEY ou use --key-file")
    return key


def build_body(model: str, chunk: str, tipo: str, ancorado: bool, seed: int,
               max_tokens: int, schema: str, temperature: float, top_p: float,
               top_k: int, presence: float) -> dict:
    """Mesmo prompt do gerador; o que muda é o transporte (OpenRouter)."""
    body = g.build_body(g.Config(teacher_model=model), chunk, tipo, ancorado, seed)
    body.pop("chat_template_kwargs", None)          # é do llama.cpp, não do OR
    body["reasoning"] = {"enabled": False}          # thinking OFF é obrigatório
    body["max_tokens"] = max_tokens
    body["temperature"] = temperature
    body["top_p"] = top_p
    body["top_k"] = top_k
    body["presence_penalty"] = presence
    if schema == "json_object":
        body["response_format"] = {"type": "json_object"}
    return body


def call(url: str, body: dict, key: str, timeout: float) -> tuple[dict, dict, str, int]:
    """Uma chamada. Devolve (json_ou_none, usage, erro, http_status)."""
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}",
                 "HTTP-Referer": "https://localhost/pingo-pilot",
                 "X-Title": "pingo-sft-pilot"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:400]
        return None, {}, f"http:{exc.code}:{detail}", exc.code
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return None, {}, f"http:{type(exc).__name__}", 0
    usage = payload.get("usage") or {}
    usage["latency_s"] = round(time.time() - t0, 2)
    usage["provider"] = payload.get("provider")
    msg = (payload.get("choices") or [{}])[0].get("message") or {}
    usage["reasoning_tokens"] = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    content = msg.get("content") or ""
    if not content and msg.get("reasoning"):
        # thinking ligado apesar do pedido: registra para não confundir com falha de formato
        usage["so_reasoning"] = True
    try:
        return json.loads(content), usage, "", 200
    except json.JSONDecodeError:
        usage["raw"] = content[:600]
        return None, usage, "json_invalido", 200


def attempt_chunk(model: str, chunk, tipo: str, ancorado: bool, cfg, key: str,
                  schema: str, args) -> list[dict]:
    """Espelha gen_sft.process_chunk: até cfg.tries tentativas; para no 1º aceito."""
    out: list[dict] = []
    for attempt in range(cfg.tries):
        seed = cfg.seed + attempt
        body = build_body(model, chunk.text, tipo, ancorado, seed, args.max_tokens,
                          schema, args.temperature, args.top_p, args.top_k,
                          args.presence_penalty)
        qa, usage, err, status = call(OR_URL, body, key, args.timeout)
        rec = {
            "model": model, "doc_id": chunk.doc_id, "chunk_idx": chunk.idx,
            "tipo": tipo, "ancorado": ancorado, "attempt": attempt,
            "erro_http": err if status and status != 200 else "",
            "motivo": "", "raw": usage.pop("raw", ""),
            "usage": usage, "chunk_chars": len(chunk.text), "url": chunk.url,
            "chunk_text": chunk.text,
        }
        if qa is None:
            rec["motivo"] = err or "sem_json"
            out.append(rec)
            if status not in (200, 0) and status != 200:
                break          # erro de API não melhora repetindo na mesma hora
            continue
        motivo, pergunta, resposta, evidencia = g.validate(qa, chunk.text, tipo, ancorado)
        rec.update(motivo=motivo or "ok", pergunta=pergunta, resposta=resposta,
                   evidencia=evidencia, qa=qa)
        out.append(rec)
        if not motivo:
            break
    return out


def run_model(model: str, chunks, args, key: str, out_dir: Path) -> dict:
    cfg = g.cfg_from_args(argparse.Namespace(
        workers=args.workers, tries=3, seed=args.seed, ancorado_frac=0.5,
        recusa_frac=0.05, chunk_chars=2000, chunk_min_chars=700, max_per_doc=3,
        domain_cap_frac=0.05, domain_cap_min=3, min_chars=2000, min_paras=4,
        max_digits_per_word=0.25, max_upper_ratio=0.3, max_urls=4, max_tokens=400,
        teacher_url=OR_URL, teacher_model=model, timeout=args.timeout,
        deny_url_regex=g.DENY_URL_DEFAULT, deny_text_regex=g.DENY_TEXT_DEFAULT))
    schema = "json_schema"
    t0 = time.time()
    seen_q: set[str] = set()
    lock = threading.Lock()
    path = out_dir / f"{slug(model)}.jsonl"
    fh = path.open("w", encoding="utf-8")

    def work(item):
        chunk, tipo, ancorado = item
        return attempt_chunk(model, chunk, tipo, ancorado, cfg, key, schema, args)

    n_accepted = 0
    idx = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # janela deslizante: sempre `workers` requisições em voo (o pool.map em lote
        # esperava o mais lento de cada lote e subestimava a vazão)
        pending: dict[Future, tuple] = {}
        while (idx < len(chunks) and idx < args.max_chunks and n_accepted < args.n) or pending:
            while (len(pending) < args.workers and idx < len(chunks)
                   and idx < args.max_chunks and n_accepted < args.n):
                pending[pool.submit(work, chunks[idx])] = chunks[idx]
                idx += 1
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                pending.pop(fut, None)
                for rec in fut.result():
                    ok = rec["motivo"] == "ok"
                    if ok:
                        with lock:
                            k = g.normalize_ws(rec["pergunta"]).lower()
                            if k in seen_q:
                                rec["motivo"] = "pergunta_duplicada"
                                ok = False
                            else:
                                seen_q.add(k)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if ok:
                        n_accepted += 1
                    if rec["motivo"].startswith("http:404") and schema == "json_schema":
                        schema = "json_object"
                        print(f"  ! {model} não aceitou json_schema, caindo para json_object")
            fh.flush()
    if n_accepted:
        print(f"  {model}: {n_accepted}/{args.n} aceitos em {idx} trechos,"
              f" {time.time() - t0:.0f}s ({n_accepted / (time.time() - t0):.2f} par/s)")
    fh.close()
    elapsed = time.time() - t0

    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    acc = [r for r in rows if r["motivo"] == "ok"]
    motivos: dict[str, int] = {}
    for r in rows:
        motivos[r["motivo"]] = motivos.get(r["motivo"], 0) + 1
    tok_in = sum(r["usage"].get("prompt_tokens", 0) for r in rows)
    tok_out = sum(r["usage"].get("completion_tokens", 0) for r in rows)
    lat = sorted(r["usage"].get("latency_s", 0) for r in rows if r["usage"].get("latency_s"))
    stats = {
        "model": model, "schema_usado": schema, "aceitos": len(acc),
        "chamadas": len(rows), "trechos": idx, "segundos": round(elapsed, 1),
        "s_por_par": round(elapsed / max(len(acc), 1), 1),
        "prompt_tokens": tok_in, "completion_tokens": tok_out,
        "prompt_tok_por_chamada": round(tok_in / max(len(rows), 1)),
        "completion_tok_por_chamada": round(tok_out / max(len(rows), 1)),
        "lat_mediana_s": round(lat[len(lat) // 2], 2) if lat else None,
        "lat_p90_s": round(lat[int(len(lat) * 0.9)], 2) if lat else None,
        "motivos": motivos,
        "providers": sorted({r["usage"].get("provider") for r in rows if r["usage"].get("provider")}),
        "reasoning_tokens": sum(r["usage"].get("reasoning_tokens") or 0 for r in rows),
    }
    (out_dir / f"{slug(model)}.stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=10, help="pares ACEITOS por modelo")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--max-chunks", type=int, default=40, help="teto de trechos por modelo")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=999, help="seed da AMOSTRA (separada do run real)")
    p.add_argument("--chunk-seed", type=int, default=999)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--presence-penalty", type=float, default=1.5)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--key-file", default="~/.openrouter_key")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out-dir", default="data/sft/or-pilot")
    args = p.parse_args(argv)
    key = load_key(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = g.cfg_from_args(argparse.Namespace(
        workers=args.workers, tries=3, seed=args.chunk_seed, ancorado_frac=0.5,
        recusa_frac=0.05, chunk_chars=2000, chunk_min_chars=700, max_per_doc=3,
        domain_cap_frac=0.05, domain_cap_min=3, min_chars=2000, min_paras=4,
        max_digits_per_word=0.25, max_upper_ratio=0.3, max_urls=4, max_tokens=400,
        teacher_url=OR_URL, teacher_model="x", timeout=args.timeout,
        deny_url_regex=g.DENY_URL_DEFAULT, deny_text_regex=g.DENY_TEXT_DEFAULT))

    # ignora trechos que o run principal já usou (o arquivo pode estar crescendo)
    done: set[tuple] = set()
    main_out = Path("data/sft/sft-v1.jsonl")
    if main_out.exists():
        for line in main_out.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["doc_id"], r["chunk_idx"]))
            except json.JSONDecodeError:
                continue

    rng = random.Random(args.seed)
    items = []
    need = args.max_chunks * max(1, len(args.models))
    for chunk, _ in g.iter_docs(Path(args.raw_dir), cfg):
        if (chunk.doc_id, chunk.idx) in done:
            continue
        tipo = g.pick_tipo(rng, cfg.recusa_frac)
        ancorado = rng.random() < cfg.ancorado_frac
        items.append((chunk, tipo, ancorado))
        if len(items) >= args.max_chunks:
            break
    print(f"{len(items)} trechos amostrados (seed {args.chunk_seed}); "
          f"os mesmos para os {len(args.models)} modelos\n")

    results = []
    for model in args.models:
        print(f"== {model}")
        results.append(run_model(model, items, args, key, out_dir))
        print()

    print(f"{'modelo':34s} {'aceitos':>7s} {'cham.':>6s} {'s/par':>6s} "
          f"{'tok in':>8s} {'tok out':>8s} {'lat med':>8s}")
    for s in results:
        print(f"{s['model']:34s} {s['aceitos']:7d} {s['chamadas']:6d} "
              f"{s['s_por_par']:6.1f} {s['prompt_tokens']:8d} {s['completion_tokens']:8d} "
              f"{(s['lat_mediana_s'] or 0):8.1f}")
    (out_dir / "resumo.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
