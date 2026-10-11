# Evaluation note: sole-selector JEV direction decisions

Four-case live evaluation (RTL, airspeed, TECS, takeoff; 40 selections,
all JEV-made, zero fallbacks) produced four validated high-confidence
reports with commit-pinned citations and numeric checks. Every executed
command passed through exactly one audited selector decision; no
bypass was observed. Uncapped consultations were load-bearing
(airspeed used 18 votes).

Cost discipline held: per-request usage, per-generation provider costs,
and key-counter deltas were recorded separately, with two known
unresolved discrepancies left unresolved rather than reconciled.

Limitations: single attempt per case; no counterfactual (unguided Luna
was not re-run on these logs); confidence/margins are recorded, never
scored; plotting was never proposed, a proposal-stage omission with
confirmation-only cost. Savings and superiority are unproven by design
of this evaluation.

Recommended cleanup (prioritized, uncommitted first/checkpoint work
only — sole behavior must stay green throughout):

1. Remove the checkpoint driver experiment (`checkpoint.py`,
   `test_checkpoint.py`, analyzer checkpoint branch, checkpoint CLI
   mode/docs) — superseded by sole-selector evidence; design decision.
2. Decide first-mode fate (forced `tool_choice` path + tests + docs):
   unproven value, retains SDK-coupling risk; either remove or justify
   with a live use case — design decision.
3. Deduplicate vote classifiers (`sole_selector._validate_vote_answer`
   vs `jev_consult._evaluate_vote_answer`) and response-cost helpers
   into one tested helper — routine cleanup.
4. Mention plotting in sole instructions — the 40-decision omission
   traces to proposal guidance, not the executor; doc fix.
5. Collapse dual metadata fields (`requested_jev_consult` bool +
   `requested_jev_mode` string) — nit.
