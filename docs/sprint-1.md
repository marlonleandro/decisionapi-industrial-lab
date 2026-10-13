# Sprint 1 — First PCB Decision

## Hypothesis

OpenAI Decisions API can assign a meaningful probability to the presence of a
visible anomaly in a VisA `pcb1` image without task-specific training.

## Controlled scope

Sprint 1 deliberately uses only one `predicate`:

`visible_defect`

We do not yet convert the answer into `PASS`, `MANUAL_REVIEW` or `REJECT`.

Reason: the first objective is to validate the visual classification signal
against known VisA ground truth before defining workflow thresholds.

## Evidence

Input:

- One PCB image.
- A short context statement saying it is an industrial PCB inspection.

Question:

- Is there a visible manufacturing anomaly or defect?

Output:

- `probability` from 0 to 1.

## Initial validation

Run one known-normal image and one known-anomalous image.

Do not draw conclusions about accuracy from two samples.

## Exit criteria

Sprint 1 is complete when:

- VisA is downloaded.
- `pcb1` is found.
- One normal and one anomalous image are prepared.
- Both calls to Decisions API execute successfully.
- Raw JSON responses are persisted.
- API key is not committed.
