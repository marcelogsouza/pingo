"""Smoke test mínimo do Passo 1: dependências importam e o pacote carrega."""

import torch


def test_pacote_importa():
    import pingo  # noqa: F401


def test_torch_importa():
    assert isinstance(torch.__version__, str)
