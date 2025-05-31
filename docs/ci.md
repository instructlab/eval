# CI for InstructLab Eval

Before running any testing locally, ensure you have run `pip install -r requirements-dev.txt` in your environment.

## Unit tests

Unit tests are designed to test specific Eval components or features in isolation. Generally, new code should be adding or modifying unit tests.

All unit tests currently live in the `tests/` directory and are run with [pytest](https://docs.pytest.org/) via [tox](https://tox.wiki/).

To run the unit tests, you can run `tox -e unit` or `tox -e unitcov` if you want to generate coverage metrics as well.

In CI, the tests are run with Python 3.11 on Ubuntu and MacOS runners - you can see the details [here](https://github.com/instructlab/eval/blob/main/.github/workflows/test.yml)

## Functional tests

Functional tests are designed to test Eval components or features in tandem, but not necessarily as part of a complex workflow. New code may or may not need a functional test but should strive to implement one if possible.

The functional test script is Shell-based and can be found at `scripts/functional-tests.sh`.

To run the functional tests, you can run `tox -e functional`.

In CI, the tests are run with Python 3.11 on Ubuntu and MacOS runners - you can see the details [here](https://github.com/instructlab/eval/blob/main/.github/workflows/test.yml)

## End-to-end (E2E) tests

InstructLab Eval has several end-to-end jobs that run to ensure compatibility with the [InstructLab Core](https://github.com/instructlab/instructlab) project.
You can see details about the types of jobs being run in the matrix below.

For more details about the E2E scripts themselves, see [the InstructLab Core documentation](https://github.com/instructlab/instructlab/blob/main/docs/maintainers/ci.md#end-to-end-e2e-tests).

### Current E2E Jobs

| Name | T-Shirt Size | Runner Host | Instance Type | OS | GPU Type | Script | Flags | Runs when? | Slack/Discord reporting? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [`e2e-nvidia-l4-x1.yml`](https://github.com/instructlab/sdg/blob/main/.github/workflows/e2e-nvidia-l4-x1.yml) | Medium | AWS |[`g6.8xlarge`](https://aws.amazon.com/ec2/instance-types/g5/) | CentOS Stream 9 | 1 x NVIDIA L4 w/ 24 GB VRAM | `e2e-ci.sh` | `m` | Pull Requests, Push to `main` or `release-*` branch | No |
| [`e2e-nvidia-l40s-x4.yml`](https://github.com/instructlab/sdg/blob/main/.github/workflows/e2e-nvidia-l40s-x4.yml) | Large | AWS |[`g6e.12xlarge`](https://aws.amazon.com/ec2/instance-types/g6e/) | CentOS Stream 9 | 4 x NVIDIA L40S w/ 48 GB VRAM (192 GB) | `e2e-ci.sh` | `l` | Manually by Maintainers, Automatically against `main` branch at 4PM UTC | Yes |

### Discord/Slack reporting

Some E2E jobs send their results to the channel `#e2e-ci-results` via the `Son of Jeeves` bot in both Discord and Slack. You can see which jobs currently have reporting via the "Current E2E Jobs" table above.

In Slack, this has been implemented via [the official Slack GitHub Action](https://github.com/slackapi/slack-github-action?tab=readme-ov-file#technique-2-slack-app).
In Discord, we use [actions/actions-status-discord](https://github.com/sarisia/actions-status-discord) and the built-in channel webhooks feature.

### Triggering an E2E job via GitHub Web UI

For the E2E jobs that can be launched manually, they take an input field that
specifies the PR number or git branch to run them against. If you run them
against a PR, they will automatically post a comment to the PR when the tests
begin and end so it's easier for those involved in the PR to follow the results.

1. Visit the [Actions tab](https://github.com/instructlab/eval/actions).
2. Click on one of the E2E workflows on the left side of the page.
3. Click on the `Run workflow` button on the right side of the page.
4. Enter a branch name or a PR number in the input field.
5. Click the green `Run workflow` button.

> [!NOTE]
> Only users with "Write" permissions to the repo can run CI jobs manually
