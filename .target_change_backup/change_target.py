from pathlib import Path
import json
import sys
import difflib

ROOT = Path.cwd()
BACKUP = ROOT / ".target_change_backup"
LOG = ROOT / "target_changes.diff"
MANIFEST = BACKUP / "manifest.json"

OLD = "0.0001"
NEW = "0.02"

def apply():
    if MANIFEST.exists():
        print("Existing backup found. Restore before applying again.")
        return

    changes = []
    patches = []

    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".yaml", ".yml"}:
            continue
        if BACKUP in path.parents:
            continue

        original = path.read_text(encoding="utf-8")
        if OLD not in original:
            continue

        updated = original.replace(OLD, NEW)
        relative = path.relative_to(ROOT)

        backup_path = BACKUP / relative
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(original, encoding="utf-8")

        path.write_text(updated, encoding="utf-8")

        changes.append(str(relative))

        patch = difflib.unified_diff(
            original.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile=f"a/{relative}",
            tofile=f"b/{relative}",
        )
        patches.extend(patch)

        print(f"[CHANGED] {relative}")

    BACKUP.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(changes, indent=2))
    LOG.write_text("".join(patches))

    print(f"\nModified {len(changes)} files.")
    print(f"Review log: {LOG}")

def restore():
    if not MANIFEST.exists():
        print("No backup manifest found.")
        return

    changes = json.loads(MANIFEST.read_text())

    for name in changes:
        source = BACKUP / name
        destination = ROOT / name

        if source.exists():
            destination.write_bytes(source.read_bytes())
            print(f"[RESTORED] {name}")

    print("\nRestoration complete.")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        restore()
    else:
        apply()
