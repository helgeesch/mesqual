# Changelog

Below the section "Release" comes one section per release, newest first. The changes released with 2.0.0 have no entry: the entries start with the first change after the 2.0.0 release.

## Version rules

mesqual follows semantic versioning. The public API is a contract for downstream packages: the public names and the documented hook methods of datasets and interpreters.

Each change of the public API and each bug fix gets an entry with its bump in the section "## Unreleased", directly below the section "Release". A pull request adds that section if it is missing. The bump per change is:

- **patch**: no observable change for code that uses the documented API correctly (crash fixes, error messages, log levels).
- **minor**: new API, or changed results where the old result contradicts the docs or is plainly wrong.
- **major**: removed or renamed API, or changed behavior that correct code can rely on.

At a release, the heading "## Unreleased" becomes `## <version> (<YYYY-MM-DD>)`, with the new version and the date of the release change. The new version is the last release raised by the highest bump of its entries: 2.1.3 with a minor and a patch entry gives 2.2.0. A release without entries is a patch release; its release change adds the heading `## <version> (<YYYY-MM-DD>)` with no entries. The same change sets the new version in `pyproject.toml`, the only version source. Pull requests do not change the version otherwise.

### Entry format

```markdown
## Unreleased

### API changes

| Change | Bump |
|---|---|
| ... | minor |

### Bug fixes

| Change | Bump |
|---|---|
| ... | patch |
```

An API change entry says what existing code changes to; for a major bump, it names the replacement. A bug fix entry says where the bug was, what it did, and what changes for existing code.

## Release

CI (`.github/workflows/ci.yml`) runs on pull requests and on pushes to `main`. For a release, the maintainer merges the change that names the version (see Version rules), sets the tag `v<version>` with the version in `pyproject.toml` on helgeesch/mesqual, and publishes a GitHub release there. `.github/workflows/release.yml` then uploads the package to PyPI after approval in the environment `pypi`, and `.github/workflows/docs.yml` deploys docs.mesqual.io.
