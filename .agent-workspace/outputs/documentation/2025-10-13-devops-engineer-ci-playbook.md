# CI/CD Playbook – Beverly Knits ERP v2

**Author:** devops-engineer  
**Date:** 2025-10-13T09:05:00Z

## Pipeline Overview
- **Lint & Test:** Installs dependencies, runs flake8, executes pytest with coverage, and enforces a minimum 80% threshold before producing artifacts.
- **Docker Build & Smoke:** Uses Buildx to build the refreshed container, stores the image as a GitHub artifact for downstream environments.
- **Deploy (Manual Approval):** Protected `production` environment requires human approval prior to execution; job emits deployment instructions referencing this playbook.

## Manual Approval Process
1. Reviewer verifies lint/test job results and ensures coverage threshold meets policy.
2. Confirm Docker artifact integrity via GitHub Actions artifact logs.
3. Approve the `production` environment gate from the Actions UI to unlock deployment job.
4. Post-approval, operate according to the rollout steps below.

## Rollout Steps
1. Download the `beverly-knits-erp-ci-image` artifact or rebuild locally using the same GitHub workflow commands.
2. Load the tarball: `docker load -i beverly-knits-erp-ci.tar`.
3. Tag and push to your registry (example): `docker tag beverly-knits-erp:ci ghcr.io/org/beverly-knits-erp:latest`.
4. Deploy via infrastructure tooling (e.g., `docker compose up -d` or Kubernetes Helm charts).
5. Run `scripts/system_health_check.py --host http://service-host --port 5006` to verify live data connectivity.

## Rollback Procedure
1. Identify the last known good image (previous tag stored in registry or artifact history).
2. Re-deploy prior version using the same deployment mechanism (compose, helm, etc.).
3. Run the health check script to ensure the reverted version is serving eFab data correctly.
4. Document the rollback in the incident log and update the gap register with root cause notes.

## Additional Notes
- Coverage threshold can be adjusted in the workflow but should remain >=80% until test reorganization lifts confidence.
- SonarQube integrations can be reintroduced as a separate stage that depends on `lint-and-test` outputs.
- Integration with registry credentials can be added via `docker/login-action` once secrets are provisioned.
