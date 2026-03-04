# Contributing to BibliophileAI

Thank you for your interest in contributing to BibliophileAI. This document explains how to get set up, where to contribute, and how to submit changes.

---

## Table of contents

- [Code of conduct](#code-of-conduct)
- [How can I contribute?](#how-can-i-contribute)
- [Development setup](#development-setup)
- [Workflow: fork, branch, commit, PR](#workflow-fork-branch-commit-pr)
- [Code and documentation standards](#code-and-documentation-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Questions](#questions)

---

## Code of conduct

- Be respectful and inclusive in discussions and reviews.
- Focus on constructive feedback and the project’s goals.
- This project follows the [MIT License](LICENSE); by contributing you agree that your contributions will be licensed under the same terms.

---

## How can I contribute?

We welcome contributions in these areas:

| Area | Examples |
|------|----------|
| **Backend services** | New or improved algorithm modules, feature engineering, event streaming (e.g. clickstream, Kafka), API endpoints, error handling. |
| **Machine learning** | Model implementations, hyperparameter tuning, evaluation metrics (NDCG, CTR, diversity), training pipelines, Feast feature definitions. |
| **Frontend** | React components, UX/accessibility, responsive layout, theme and styling, integration with new API endpoints. |
| **Infrastructure** | Docker/Kubernetes manifests, CI/CD, monitoring (Prometheus/Grafana), ServiceMonitors, dashboards. |
| **Documentation** | API docs, architecture diagrams, SETUP/README updates, tutorials, inline docstrings. |
| **Data and tooling** | Book import scripts, Feast config, Airflow DAGs, dev scripts. |

**Good first issues:** Look for issues labeled `good first issue` or `help wanted`. Small doc fixes, tests, or config improvements are a great way to start.

**Before large changes:** Open an issue or discussion to align on approach and scope (e.g. new algorithm, new service, breaking API change).

---

## Development setup

1. **Clone your fork** (after forking the repo on GitHub):
   ```bash
   git clone https://github.com/YOUR_USERNAME/BibliophileAI.git
   cd BibliophileAI
   ```

2. **Add upstream** (optional, for syncing with main repo):
   ```bash
   git remote add upstream https://github.com/RAAHUL-tech/BibliophileAI.git
   ```

3. **Follow [SETUP.md](SETUP.md)** to run the stack locally (Kind, Redis, secrets, services, ingress, frontend). You don’t need Airflow or training jobs for most backend/frontend work.

4. **Per-component dev:**
   - **Python services:** See `src/<service>/README.md` and use the service’s `requirements.txt`; run with uvicorn or the Dockerfile.
   - **Frontend:** `cd frontend/bibliophile-ai-frontend && npm install && npm run dev`.
   - **Training jobs:** See `src/model_training_service/` and subfolder READMEs; run with Ray locally or via Kubernetes/Airflow.

---

## Workflow: fork, branch, commit, PR

1. **Fork** the repository on GitHub and clone your fork (see above).

2. **Create a branch** from the default branch (e.g. `main`):
   ```bash
   git checkout main
   git pull upstream main   # if you added upstream
   git checkout -b feature/your-feature-name
   # or: fix/docs-typo, refactor/cleanup-xyz
   ```

3. **Make your changes.** Keep commits focused and messages clear:
   ```bash
   git add .
   git commit -m "feat(recommendation): add diversity threshold to LTR postprocess"
   ```

4. **Push** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a pull request** against `RAAHUL-tech/BibliophileAI` (default branch). In the PR:
   - Describe what changed and why.
   - Reference any related issues (e.g. “Fixes #123”).
   - Note any setup or config changes (e.g. new env vars, new dependencies).
   - Confirm you’ve run relevant checks (lint, tests, manual smoke test if applicable).

6. **Address review feedback** by pushing new commits to the same branch. Avoid force-pushing once the PR is under review unless the maintainer prefers it.

7. **After merge:** You can delete your branch and pull the latest `main` from upstream.

---

## Code and documentation standards

- **Python:** Follow PEP 8; use type hints where practical. Keep module-level docstrings (see existing files) for new modules. Prefer `Optional[...]` for Python 3.9 compatibility where the project supports it.
- **TypeScript/React:** Use the project’s existing patterns (functional components, hooks). Run `npm run lint` in the frontend before submitting.
- **YAML/Kubernetes:** Use spaces; keep keys and naming consistent with existing manifests (e.g. labels, port names).
- **Commits:** Prefer clear, present-tense messages (“Add X” / “Fix Y”). Optionally use a short prefix like `feat:`, `fix:`, `docs:`, `refactor:`.
- **Docstrings:** New or modified Python modules and important functions should have a brief docstring; new README or SETUP sections should be accurate and concise.

---

## Testing

- **Backend:** Add or extend tests when you add behavior (e.g. unit tests for new recommendation logic, API tests for new endpoints). Run existing tests before submitting.
- **Frontend:** Run `npm run build` to ensure the app builds; run the linter (`npm run lint`). Add tests for new components or flows if the project has a test runner configured.
- **Integration:** For larger changes, manually verify key flows (e.g. login → recommendations, search, bookmark) using the steps in [SETUP.md](SETUP.md).

---

## Documentation

- Update **README.md** if you change high-level behavior, architecture, or “Getting Started” steps.
- Update **SETUP.md** if you add or change setup steps, env vars, or required services.
- Update the **README in the affected folder** (e.g. `src/recommendation_service/README.md`, `kubernetes/monitoring/README.md`) when you change that component’s behavior or usage.
- Keep **CONTRIBUTING.md** in mind when adding contribution-related notes elsewhere.

---

## Questions

- **Bugs and features:** Open a [GitHub Issue](https://github.com/RAAHUL-tech/BibliophileAI/issues). Use the issue template if one is provided.
- **Ideas and design:** Open a Discussion or an Issue with a clear title and description so maintainers and others can weigh in.

Thank you for contributing to BibliophileAI.
