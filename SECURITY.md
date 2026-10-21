# Security Policy

## Scope

This project is a developing hackathon repository. Security controls should be proportionate to the actual implementation and data sensitivity.

## Reporting a vulnerability

Do not publish credentials, tokens, personal data, or exploit details in a public issue. Contact the repository owner through a private GitHub communication channel and provide a concise description, affected revision/component, reproduction steps, impact, and suggested mitigation where known.

## Secret handling

- Never commit API keys, access tokens, passwords, private keys, production URLs containing credentials, or secret-bearing logs.
- Use environment variables or an appropriate secret store.
- Keep `.env` files and local credential files out of version control.
- Commit placeholders only, never real values.
- Rotate a credential immediately if it is exposed; deleting the file in a later commit does not erase it from Git history.
- Use least-privilege tokens and separate development from production credentials.

## Data protection

Do not ingest or publish personal, sensitive, restricted, or non-redistributable data unless its use is necessary, authorized, and documented. Minimize retention and document third-party data flows where applicable.

## Dependencies and external services

Use supported versions, lock dependencies where possible, review license terms, and configure network timeouts and quotas. Avoid exposing raw exception traces or sensitive payloads to end users.

## Incident response

1. Stop further exposure and revoke/rotate affected credentials.
2. Determine affected revisions, services, data, and access.
3. Remove or restrict exposed material and remediate the underlying cause.
4. Review Git history and downstream deployments for persistence.
5. Record the incident and prevention measures without publishing secrets.
