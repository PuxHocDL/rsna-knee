# Model comparison (2026-10-18)

gold58 = the 58 radiologist-labelled train studies (no model trained on them). ref392 = 392 train studies with zoliseress reference labels
(evaluation only); our models never trained on them (label v5h holds them out). Public models were trained on the ref392 studies,
so only R6 out-of-fold is a fair public reference there. Macro AUC over the 12 findings; per-finding values are in the CSV.

| model | kind | gold58 | ref392 | LB_solo | note |
|---|---|---|---|---|---|
| R6 4-fold + flip TTA | public | 0.9366 |  |  | trained on these studies (in-sample), not measurable on ref392 |
| R6 out-of-fold (1 fold per study) | public |  | 0.9155 |  | fair on ref392 |
| R7 4-fold | public | 0.9317 |  |  | trained on these studies (in-sample), not measurable on ref392 |
| S6fall | public | 0.9314 |  | 0.950 | trained on these studies (in-sample), not measurable on ref392 |
| Raptor 384 | public | 0.9229 |  | 0.949 | trained on these studies (in-sample), not measurable on ref392 |
| Raptor 224 | public | 0.915 |  | 0.945 | trained on these studies (in-sample), not measurable on ref392 |
| B fold 0 (v3) | ours | 0.9266 | 0.9166 |  |  |
| B fold 1 (v4h) | ours | 0.9289 | 0.9219 |  |  |
| B fold 2 (v5h) | ours | 0.9323 | 0.9281 |  |  |
| B fold 3 (v5h) | ours | 0.9249 | 0.9265 |  |  |
| B fold 4 (v5h) | ours | 0.9298 | 0.9264 |  |  |
| B fold 5 (v5h R2) | ours | 0.9216 | 0.914 |  |  |
| B fold 6 (v5h R2) | ours | 0.9209 | 0.9183 |  |  |
| B fold 7 (v5h) | ours | 0.9261 | 0.9275 |  |  |
| B4 = folds 2,3,4,7 (v5h) | ours | 0.9307 | 0.929 | pending |  |
| B5 = fold 1 + folds 2,3,4,7 | ours | 0.9323 | 0.9286 | 0.950 |  |
| B2F = folds 0,1 | ours | 0.9312 | 0.9235 | 0.947 |  |
| B all 8 folds | ours | 0.9345 | 0.9279 |  |  |
| CoAtNet-2 v4h | ours | 0.9198 | 0.9174 |  |  |
| CoAtNet-2 v4h + train mirror | ours | 0.9284 | 0.9166 |  |  |
| CoAtNet-2 v5h | ours | 0.924 | 0.9195 |  |  |
| CoAtNet-2 v5h + train mirror | ours | 0.9217 | 0.923 |  |  |
