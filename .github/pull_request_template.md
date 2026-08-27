## Summary

<!-- What does this PR change and why? -->

## Changes

<!-- Bullet the key changes. -->
-

## Testing

<!-- How did you verify this? -->
- [ ] `make check` passes locally (black + pylint + tests)
- [ ] Added/updated tests for the change
- [ ] `python main.py --visits 2000 --model random_forest` runs clean

## Checklist

- [ ] Tests are offline and deterministic (no network; RNGs seeded)
- [ ] Optional/heavy dependencies (if any) are imported lazily and degrade gracefully
- [ ] Docs updated (`README.md` / `docs/`) where relevant
- [ ] `CHANGELOG.md` updated under **Unreleased**
- [ ] No secrets committed; runtime config comes from the environment
