#!/usr/bin/env bash
# Install repo git hooks (tracked in scripts/hooks/) into .git/hooks/.
# Run once after cloning:  scripts/install-hooks.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

install -m 0755 scripts/hooks/pre-push .git/hooks/pre-push
echo "✓ installed .git/hooks/pre-push"

# Retire the old export-on-commit hook if present.
if [ -f .git/hooks/pre-commit ] && grep -q "export_and_clean.py" .git/hooks/pre-commit 2>/dev/null; then
  mv .git/hooks/pre-commit .git/hooks/pre-commit.retired
  echo "✓ retired old .git/hooks/pre-commit -> .git/hooks/pre-commit.retired"
fi
echo "Done."
