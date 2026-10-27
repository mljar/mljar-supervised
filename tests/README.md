# Running tests


To run all tests:

```
pytest tests -v -x
```

To run tests for `algorithms`:

```
pytest tests/tests_algorithms -v -x -s
```

For opt-in performance and RAM benchmarks, see [performance/README.md](performance/README.md).
Initial measured bottlenecks are documented in [performance/FINDINGS.md](performance/FINDINGS.md).
