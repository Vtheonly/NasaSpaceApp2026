# Official Submission Checklist (issues #1 and #17)

Official details below were verified from NASA Space Apps materials on
**2026-10-22**. Re-verify anything time-sensitive before submitting; the
Sources section records the exact URLs and what each supports.

## 1. Event facts (verified 2026-10-22)

| Item | Verified value | Source |
|---|---|---|
| Event | 2026 NASA Space Apps Challenge, Saturday **Nov 14 - Sunday Nov 15, 2026** | spaceappschallenge.org homepage + 2026 event page |
| Registration | Opened Aug 26, 2026; open until the hackathon closes Nov 15 | 2026 event page timeline |
| Team size | **No more than six participants**; form/join via Team Finder (open since Sep 17) | 2026 event page; Team Formation Guide |
| Local event | Every member must join the same Local Event (or the virtual Universal Event) to be eligible for Global Judging | Universal Event page; Team Formation Guide |
| Submission window | Opens Sat Nov 14, 9:00 AM; closes **Sun Nov 15, 11:59 PM local time**; submit via the Project tab on the team page | 2026 event page timeline; Universal Event page |
| Participation terms | Participant Terms and Conditions (updated Aug 7, 2026): personal capacity, minors need guardian registration/accompaniment, U.S. federal civil servants ineligible, China-affiliation participation restrictions | /legal |
| Judging criteria | Five criteria: **Impact, Creativity, Validity, Relevance, Presentation** (Project Submission Guide) | Resources page guides |
| Participant survey | Complete by Nov 30, 2026 | 2026 event page timeline |

Challenge statements for 2026 (14 challenges) are live on the event site; the
team's chosen challenge is **"Harmonization of MODIS and VIIRS Hot Spots"**
(owner-supplied prompt, recorded in docs/CHALLENGE_AND_REQUIREMENTS.md).
Action: confirm the official challenge page wording matches the supplied
prompt when statements are re-checked before the event (the site listed
challenge summaries as live at check time).

## 2. Submission checklist (team-owned; verify per item)

### Registration and team
- [ ] Every member has a 2026 spaceappschallenge.org account and is **registered**.
- [ ] Team has **<= 6 members**, all in the **same Local Event / Universal Event**.
- [ ] Team page lists the challenge "Harmonization of MODIS and VIIRS Hot Spots".

### Project page content
- [ ] Project name, short description, and full description written (English).
- [ ] Demo video/presentation prepared (Presentation criterion); rehearse via
      docs/DEMO_GUIDE.md, including the failure drill.
- [ ] Links: GitHub repository, demo/deployed artifact (if any), data sources.
- [ ] Photo/images and any required fields per the current Project Submission
      Guide on the Resources page (re-check at submission time; guides are
      event-specific).

### Scientific claims audit ("no unsupported claims" - issue #17)
- [ ] Every performance/scientific claim in the description maps to an
      artifact in the repo (docs/DEMO_GUIDE.md evidence checklist).
- [ ] Synthetic-fixture evidence is labeled SYNTHETIC everywhere it appears
      (run_demo enforces this; demo report says it verbatim).
- [ ] The description says "model estimates, not ground truth" for harmonized
      values and "experimental/uncalibrated" for forecast intervals.
- [ ] No exact-ignition-location/time claim anywhere (regional activity only).
- [ ] Real-data status is stated accurately: schema + five-pass gate verified
      on official FIRMS NRT files (2026-10-20/21 samples, see audit doc);
      full-archive harmonization/forecast validation is NOT yet claimed.

### Attribution and terms (issue #19 audit doc)
- [ ] NASA FIRMS / LANCE attribution with product names (MODIS Collection 6.x,
      VIIRS 375 m S-NPP) in README, demo, and project description.
- [ ] Dependency licenses listed (docs/DEPENDENCY_AND_SECURITY_AUDIT.md).
- [ ] Optional Prithvi/HLS usage (if included by then) carries its license and
      checkpoint hash; if not included, the description does not mention it as
      a feature.
- [ ] No credentials or large data/weights in the repo (GitGuardian clean).

### Final rehearsals
- [ ] `pytest tests` green locally and CI on the submission head commit.
- [ ] `python scripts/run_demo.py --output-dir out/demo` completes and the
      report numbers match `demo_summary.json`.
- [ ] Clean-setup run from a fresh clone (docs/DEMO_GUIDE.md section 0).
- [ ] Links in the project page actually open (repo, demo, sources).
- [ ] Submit before **Nov 15, 11:59 PM local time**; screenshot the receipt.

## 3. Blockers / open items

| Item | Status | Owner |
|---|---|---|
| Official Project Submission Guide field list (2026-specific) | Re-check on the Resources page at submission time; 2026 guides listed as available there | team |
| Official challenge page wording vs owner-supplied prompt | Verify challenge statement when re-checked before the event | team |
| Real FIRMS archive validation (beyond the verified NRT schema smoke) | Open - needs a downloaded multi-year archive (issue #6) | team |
| Deployed app URL for the project page | Optional per current guides; the offline pipeline + demo is the core deliverable (app tracked in #14 / backlog D3) | team |

## 4. Sources (checked 2026-10-22)

1. https://www.spaceappschallenge.org - event dates Nov 14-15, 2026; registration open.
2. https://www.spaceappschallenge.org/2026 - step timeline incl. submission window (Nov 14 9:00 AM - Nov 15 11:59 PM local).
3. https://www.spaceappschallenge.org/legal - 2026 Participant Terms and Conditions (updated Aug 7, 2026).
4. https://www.spaceappschallenge.org/resources/team-formation-guide - team rules, project due date.
5. https://www.spaceappschallenge.org/2026/local-events/universal-event - local-event consistency requirement for global judging.
6. https://science.nasa.gov/universe/nasa-welcomes-all-ages-experience-levels-for-space-apps-challenge - NASA confirmation of dates.
7. https://firms.modaps.eosdis.nasa.gov/data/active_fire/ - real FIRMS NRT files used for the schema verification (see docs/DEPENDENCY_AND_SECURITY_AUDIT.md).
