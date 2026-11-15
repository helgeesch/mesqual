# Use mesqual with agents

mesqual is made for humans and agents alike. This page gives the skill that ships with mesqual, a snippet for the `AGENTS.md` of your project, and [`llms.txt`](llms.txt), an index of these docs for language models. [What agent-ready means](#what-agent-ready-means) lists what makes mesqual ready for agents, with a link to each.

## What agent-ready means

- **Describe methods with JSON-ready output.** `as_records=True` gives plain dicts on [`StudyManager.describe_study`](api_reference/study_manager.md#mesqual.study_manager.StudyManager.describe_study), [`describe_datasets`](api_reference/datasets.md#mesqual.datasets.Dataset.describe_datasets) and [`describe_flags`](api_reference/datasets.md#mesqual.datasets.Dataset.describe_flags). [`dataset.flag_info(flag).as_record()`](api_reference/datasets.md#mesqual.datasets.FlagInfo.as_record) does the same for one flag. These calls load no time series data ([Describe a dataset](api_reference/datasets.md#describe-a-dataset)).
- **Offline synthetic data.** [`mesqual.testing`](api_reference/testing.md#mesqual.testing.make_dummy_study) builds a study without network access and without an external platform:

    ```python
    from mesqual.testing import make_dummy_study
    study = make_dummy_study()
    ```

- **Error messages that name the allowed values.** Examples on `dataset = study.scen.get_dataset('base')` of the synthetic study:

    ```text
    >>> dataset.fetch('Node.Results.price', solution='x')
    ValueError: Flag Node.Results.price: 'x' is not a value of 'solution'. Allowed: ['first', 'final'].
    >>> dataset.fetch('Node.Results.price', bogus=1)
    ValueError: Flag Node.Results.price: no flag spec or dataset class declares the parameters ['bogus']. Declared: ['solution'].
    ```

    A [`FlagUnavailableError`](api_reference/flag.md#mesqual.flag.FlagUnavailableError) names the flag, the dataset and the reason. An `UnknownFlagError` only says that no spec declares the flag: it does not name the fix, also when the layer in the flag name is unknown. Use `describe_flags` then. Section 10 of the skill, "When a call fails", lists the steps.
- **A versioned API with a change log.** mesqual follows semantic versioning. The public API is a contract, and each API change and bug fix after 2.0.0 has an entry with its bump in the [CHANGELOG](https://github.com/helgeesch/mesqual/blob/main/CHANGELOG.md).
- **Declared dependency floors.** [`pyproject.toml`](https://github.com/helgeesch/mesqual/blob/main/pyproject.toml) requires Python 3.11 or later, `pandas>=2.1`, `numpy>=1.23.2`, `matplotlib>=3.6` and `pint>=0.24.4`. Other dependencies have no floor. CI tests Python 3.11 to 3.14 with the latest dependencies. It does not test the dependency floors.
- **Outputs that an agent can check as text.** [`fetch`](api_reference/datasets.md#mesqual.datasets.Dataset.fetch) returns a pandas DataFrame or Series, [`KPICollection.to_dataframe()`](api_reference/kpis/collection.md#mesqual.kpis.collection.KPICollection.to_dataframe) returns a frame of KPIs. Changed text values of a comparison have text markers, see [the conventions](#snippet-for-agentsmd). The skill says to check the numbers as text before you build figures.
- **An index for language models.** [`llms.txt`](llms.txt) lists the pages of these docs, one line each.
- **A skill for the workflow.** `mesqual-study` ships with the package. See [Install the skill](#install-the-skill).

## Install the skill

mesqual ships the skill `mesqual-study` in the Agent Skills format. It gives the workflow of an analysis in a study and links into these docs. Install it once per project, and again after each upgrade of mesqual, with the same options. Run the command yourself, or give it to your agent as a task:

```bash
python -m mesqual.skills install .            # to ./.agents/skills/mesqual-study/
python -m mesqual.skills install . --claude   # also to ./.claude/skills/mesqual-study/, for Claude Code
```

The command prints one line per skill folder, with its path and `created`, `updated` or `unchanged`. The shipped files overwrite files of the same name; other files in the skill folder stay. After an upgrade, paste the snippet below again too: it holds the conventions of the installed version. Commit the skill folders (`.agents/skills/`, and `.claude/skills/` with `--claude`), so every clone has them.

## Snippet for AGENTS.md

Paste this section into the `AGENTS.md` of your project:

```markdown
## mesqual

This project analyzes energy market simulations with mesqual (https://docs.mesqual.io). The skill `mesqual-study` gives the workflow of an analysis. It is the file `.agents/skills/mesqual-study/SKILL.md` (or `.claude/skills/mesqual-study/SKILL.md`): read it if you do not load skills. If the file is missing, see https://docs.mesqual.io/use-with-agents/.

Never guess a flag name. Ask the study first. These calls load no time series data:

- `study.describe_study(as_records=True)`: the scenarios, the comparisons and the flags of each, as plain dicts.
- `study.scen.describe_datasets(max_depth=1, as_records=True)`: the scenarios with descriptions and attributes.
- `dataset.describe_flags(contains='price', as_records=True)`: the flags of one dataset, with availability and parameters.
- `dataset.flag_info(flag).as_record()`: description, unit, schema and parameters of one flag.
--8<-- "src/mesqual/skills/mesqual-study/SKILL.md:templates"

The result of `fetch`:

--8<-- "src/mesqual/skills/mesqual-study/SKILL.md:conventions"
```

## Where the conventions come from

- The levels of a result: the schema of the flag spec ([`FrameSchema` and `Level`](api_reference/flag.md#mesqual.flag.Level)), shown by [`flag_info`](api_reference/datasets.md#mesqual.datasets.FlagInfo).
- The level `dataset` and the outer join: [`DatasetConcatCollection`](api_reference/datasets.md#mesqual.datasets.DatasetConcatCollection).
- Comparisons: [`DatasetComparison`](api_reference/datasets.md#mesqual.datasets.DatasetComparison) and [`StudyManager`](api_reference/study_manager.md).
- Parameters, config and granularity of a fetch: [`Dataset.fetch`](api_reference/datasets.md#mesqual.datasets.Dataset.fetch).
- The contracts behind them: [Core contracts](architecture/core-contracts.md#3-results) and [Flags and layers](architecture/flags-and-layers.md#4-values-units-schema-topology-memberships).
