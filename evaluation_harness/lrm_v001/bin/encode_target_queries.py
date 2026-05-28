#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "proposed_2-mmoe_ple" / "infer" / "lrm_v001"))
from encode_target_queries import main
if __name__ == "__main__":
    raise SystemExit(main())
