# Maintain runnable documentation recipes

Keep user-facing code in `examples/scripts/`. These standard-library tools sync
that code into docs downloads and fenced examples; they do not train models.

```bash
python examples/docs/generate_custom_validation.py
python examples/docs/generate_custom_validation.py --check
```

Edit the grouped/time script for code changes and
`docs/docs/features/custom-validation.md` for explanations. The generator owns
the marked code blocks. Record real runs separately before updating diagrams,
metrics, or evidence under `docs/docs/assets/custom-validation/`. Preserve the
logs, exact source copies, full models, reports, and split indices under
`performance-results/`. The synthetic data here teach validation mechanics;
do not present their scores as real-domain benchmarks.
