# Dataset Validation References

A check tests one rule on one dataset and returns a `CheckResult`. A `DatasetValidator` runs a set of checks and returns a `ValidationReport`.

```python
from mesqual.validation import ConstraintCheck, DatasetValidator

validator = DatasetValidator([
    ConstraintCheck('Line.Constraints.capacity_up', min_value=0),
    ConstraintCheck('Generator.Results.generation', min_value=0, isna_ok=False),
])
report = validator.run(dataset)
report.failed                      # the FAIL results
report.to_markdown('validation.md')
```

::: mesqual.validation
