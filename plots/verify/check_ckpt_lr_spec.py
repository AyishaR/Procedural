"""usage: check_ckpt_lr_spec.py CHECKPOINT LR_JSON -- the checkpoint's recorded lr specification must equal the launch file."""
import json, sys, torch
sys.path.insert(0, "/home/schrodi/Procedural")
from row_lr_mask import split_lr_scale_spec
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sc, rows = split_lr_scale_spec(json.load(open(sys.argv[2])))
want = {"scalar": sc, "rows": {k: [list(map(float, r)) for r in v] for k, v in rows.items()}}
same = ck.get("lr_scale_spec") == want
print(f"checkpoint lr_scale_spec equals the launch file: {same} | epoch {ck.get('epoch')} | model finite: {all(bool(torch.isfinite(v).all()) for v in ck['model'].values())}")
sys.exit(0 if same else 1)
