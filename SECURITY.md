# Security and Data-Handling Policy

## Scope

This hackathon prototype processes public Earth-observation data, model artifacts, and user-selected AOI geometry. It is not a certified emergency-response system.

## Reporting

Report suspected credential exposure or unsafe data handling privately to the repository owner rather than posting live secrets in an issue. Include affected paths, impact, and safe reproduction details. Do not attach passwords, tokens, or private user data.

## Secrets and artifacts

- Never commit API keys, tokens, passwords, private keys, cookies, or credential-bearing configuration.
- Ignore local `.env` files; commit only safe placeholders in `.env.example`.
- Revoke/rotate exposed credentials and assess Git history and logs.
- Keep virtual environments, wheelhouses, model weights, raw/processed data, and generated outputs outside Git unless a small license-cleared fixture is essential.
- Verify model/data hashes and provenance; do not blindly execute untrusted notebooks, pickle files, or binaries.

## Data governance

- Respect source and model licenses/terms and preserve attribution.
- Avoid collecting personal data; the core workflow should not require identity.
- Validate AOI geometry and bound expensive requests.
- Avoid exposing filesystem paths, secrets, or sensitive environment details in UI errors.

## Offline runtime

Use staged local artifacts and fail clearly when required files are missing. Do not add hidden network downloads or telemetry to the offline pipeline. The app must disclose unavailable optional data and must not silently substitute synthetic data.

## Reliability disclaimer

The prototype provides historical analysis and probabilistic regional activity estimates. It must not imply that a forecast guarantees ignition, safety, or the absence of fire.