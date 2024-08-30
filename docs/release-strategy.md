# InstructLab Eval Release Strategy

This document discusses the release strategy and processes for the
`instructlab-eval` Python package built from the
<https://github.com/instructlab/eval> git repository.

## Versioning Scheme

Releases use a `X.Y.Z` numbering scheme.

X-stream release are for major releases. At this stage in the project a major release has not been cut and we expect each release to be a new Y-stream.

Z-stream releases are meant for critical bug and documentation fixes. Z-stream releases are cut as maintainers see fit.

## Schedule

The project currently operates on an ad-hoc release schedule based on the discretion of the maintainers team.

The cadence for major releases starting from 1.0 onward will be determined as the project matures.

A schedule will be updated in a markdown file on the <https://github.com/instructlab/eval> GitHub repository.

## Release Tracking

Currently there is no formal process of release tracking. GitHub Issues are used for tracking individual work items.

In the future, the project may use Milestones or Project Boards for more formal release planning. At that time this document will be updated.

## Git Branches and Tags

Every `X.Y` release stream gets a new branch.

Each release, `X.Y.Z`, exists as a tag named `vX.Y.Z`.

## Release Branch Maintenance

Maintenance efforts are only on the most recent Y-stream.
Critical bug fixes are backported to the most recent release branch.

## Release Mechanics

Release mechanics are done by a Release Manager identified for that release.
The Release Manager is a member of the Eval Maintainers team that has agreed to take on these responsibilities.
The Release Manager can change on a per-release basis.

The following are the steps for how Y-stream and Z-stream releases gets cut.

### Y-Stream

1. Determine a commit on the main branch that will serve as the basis for the next release - most of the time this should be the latest commit.
1. Create a new release branch in the format `release-vX.Y` off of the determined commit (will match `main` if the latest commit is chosen).
1. Create a new release on GitHub targeting the release branch and using the latest Y-Stream tag as the previous release (e.g. `0.15.1` precedes `0.16.0`).
1. Announce release via the following:
    - The `#eval` channel on Slack
    - The `dev` mailing list

### Z-Stream

1. Backport all relevant commits from `main` to the `release-vX.Y` branch.
    - It may also be the case you wish to update release branch first - if this approach is taken, ensure any relevant commits are subsequently backported to `main`
1. Create a new release on GitHub targeting the release branch and using the previous Z-Stream tag as the previous release (e.g. `0.15.0` precedes `0.15.1`).
1. Announce release via the following:
    - The `#eval` channel on Slack
    - The `dev` mailing list

## Release Notes

The project maintains a single `CHANGELOG.md` file that documents all releases. To ensure our users
are well-informed about new features, improvements, and breaking changes, we maintain a
`CHANGELOG.md` file. This file serves as a centralized place to document changes that will be
included in the next (X) or (Y) release. Given that the project is in its early stages, we are
currently focused on documenting changes for the next (Y) release.

### Editing Release Notes

When submitting a Pull Request (PR) that introduces notable features or breaking changes, committers
need to update the `CHANGELOG.md` file. Clearly describe the changes, their impact, and
any actions users might need to take. We want clear, concise, and user-friendly notes.

### Branching for a New Release

Each time we prepare for a new (X) or (Y) release, we branch out from the main codebase.
As part of this branching process, the contents of `CHANGELOG.md` are reviewed and
finalized.
