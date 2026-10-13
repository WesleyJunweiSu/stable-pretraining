"""Check that the DDP test assertions detect common distributed-gathering bugs.

Runs the exact assertions of test_global_loss_and_ddp_parameter_gradients
against the real MMCRLoss and against deliberately broken variants.
"""

import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from stable_pretraining.losses import MMCRLoss


def mutant_loss(kind, lmbda, gather):
    if kind == "real" or not gather:
        return MMCRLoss(lmbda, gather_distributed=gather)

    def fn(p):
        z = F.normalize(p, dim=-1)
        w = dist.get_world_size()
        if kind == "no_gather":
            pass
        elif kind == "detached_gather":
            # Non-autograd gather, own slice re-inserted (SimCLR-style trick).
            parts = [torch.empty_like(z) for _ in range(w)]
            dist.all_gather(parts, z.detach().contiguous())
            parts[dist.get_rank()] = z
            z = torch.cat(parts)
        elif kind == "worldsize_scaled":
            z = torch.cat(torch.distributed.nn.functional.all_gather(z.contiguous()))
        c = z.mean(1)
        g = torch.linalg.svdvals(c).sum()
        loss = -g if lmbda == 0 else lmbda * torch.linalg.svdvals(z).sum(-1).mean() - g
        return loss * w if kind == "worldsize_scaled" else loss

    return fn


def worker(rank, rendezvous, kind, out):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    result = "PASS"
    try:
        torch.manual_seed(23)
        images = torch.randn(6, 3, 5, dtype=torch.float64)
        for lmbda in (0.0, 0.2):
            enc = nn.Linear(5, 4, bias=False).double()
            ddp = DistributedDataParallel(enc)
            ref = nn.Linear(5, 4, bias=False).double()
            ref.load_state_dict(enc.state_dict())
            loss = mutant_loss(kind, lmbda, True)(ddp(images.chunk(2)[rank]))
            full = mutant_loss(kind, lmbda, False)(ref(images))
            loss.backward()
            full.backward()
            try:
                torch.testing.assert_close(loss, full, rtol=1e-9, atol=1e-9)
                torch.testing.assert_close(
                    enc.weight.grad, ref.weight.grad, rtol=1e-8, atol=1e-8
                )
            except AssertionError as e:
                result = f"FAIL(lmbda={lmbda}): {str(e).splitlines()[0]}"
                break
    finally:
        dist.destroy_process_group()
    if rank == 0:
        Path(out).write_text(result)


if __name__ == "__main__":
    # The real loss must pass and every injected bug must be caught.
    expected = {
        "real": True,
        "no_gather": False,
        "detached_gather": False,
        "worldsize_scaled": False,
    }
    ok = True
    for kind, should_pass in expected.items():
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.txt"
            mp.spawn(
                worker,
                args=((Path(d) / "rdv").as_uri(), kind, str(out)),
                nprocs=2,
                join=True,
            )
            result = out.read_text()
        caught = "expected PASS" if should_pass else "bug must be detected"
        good = result.startswith("PASS") == should_pass
        ok &= good
        print(f"{kind:18s} -> {result}   [{caught}: {'OK' if good else 'UNEXPECTED'}]")
    sys.exit(0 if ok else "Mutation check did not behave as expected")
