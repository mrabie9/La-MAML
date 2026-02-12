# delete_bad_logs.py
import argparse
import shutil
from pathlib import Path

def main(base="logs", dry_run=True):
    base = Path(base)
    deleted = []
    kept = []
    if not base.is_dir():
        print(f"{base} does not exist or is not a directory.")
        return

    # Assumes structure: logs/<method>/<run>/0/results.txt
    for method_dir in base.iterdir():
        if not method_dir.is_dir() or method_dir.name.startswith("tuning") or method_dir.name.startswith("iq_experiments"): 
            continue
        for run_dir in method_dir.iterdir():
            if not run_dir.is_dir():
                continue
            ok_file = run_dir / "0" / "results.txt"
            if ok_file.is_file():
                kept.append(run_dir)
            else:
                deleted.append(run_dir)
                if not dry_run:
                    shutil.rmtree(run_dir)

    print(f"Kept {len(kept)} run(s).")
    for p in kept:
        print(f"  KEEP: {p}")

    print(f"\n{'Would delete' if dry_run else 'Deleted'} {len(deleted)} run(s).")
    for p in deleted:
        print(f"  DELETE: {p}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="logs", help="Base logs directory")
    ap.add_argument("--yes", action="store_true", help="Actually delete (disable dry-run)")
    args = ap.parse_args()
    main(base=args.base, dry_run=not args.yes)
