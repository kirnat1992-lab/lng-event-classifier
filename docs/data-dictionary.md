# Demo dataset and labeling guide

All 120 texts are **fictional, handcrafted synthetic examples**, not verified historical cargo movements, vessel sightings, sanctions designations or news headlines. They are for learning and testing software only. No actual vessel identity, cargo, port visit, or sanction is asserted.

## Columns

| Column | Meaning |
| --- | --- |
| `event_id` | Stable fictional demo record ID (`DEMO-*`) |
| `text` | Short illustrative shipping event description |
| `label` | Exactly one category from the taxonomy below |
| `split` | Predeclared `train` (80 records) or `test` (40 records) |
| `provenance` | `SYNTHETIC` or (only for future sourced examples) `PUBLIC_VERIFIED` |
| `source_url` | Blank for synthetic data; mandatory URL for `PUBLIC_VERIFIED` examples |

## Event taxonomy

| Label | Meaning |
| --- | --- |
| `loading` | Vessel receives LNG cargo at an export facility |
| `departure` | Vessel departs a port |
| `arrival` | Vessel arrives at a port or terminal |
| `discharge` | Vessel unloads LNG at an import facility |
| `ship_to_ship_transfer` | LNG moves between vessels offshore |
| `route_diversion` | Route or destination changes |
| `delay` | Voyage or berth timing is interrupted |
| `sanctions` | Official designation or restriction |
| `detention` | A maritime authority detains or seizes vessel |
| `vessel_status` | AIS / navigational-status update |

These ten **demo** categories are mutually exclusive for the exercise. Real records may contain multiple events; future research should consider multi-label annotation, ambiguous examples, duplicated syndications, and explicit `unknown` handling. A model classification is never evidence that an event happened.

## Leakage / evaluation cautions

- Train/test assignments are fixed in the CSV; do not tune prompts or model based on the test set.
- Test descriptions are also handcrafted and often share vocabulary with training descriptions. Test scores therefore overestimate likely performance on real reports.
- For a real study, annotate independently sourced event reports, group same-source/reposted reports, and evaluate on later reports and previously unseen vessels/ports.
- Keep the gold-standard labeling process separate from predictions; record annotator agreement and disagreements.
- Avoid automatically republishing licensed news text or confidential grant-research material.
