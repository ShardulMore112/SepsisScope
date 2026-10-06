# Stage 4.0 — Final Evaluation

## Frozen candidate
**TR1 F0+F1+F2+F3**, equal-weight mean of the four Transformer representation probabilities.

Selection was based on validation AUPRC. The validation threshold was selected by maximum F1 over 1001 thresholds and then frozen for test evaluation.

**Frozen threshold:** 0.764

## Validation
- Timesteps: 182,553
- Positive timesteps: 3,334
- Timestep prevalence: 1.8263%
- AUPRC: 0.089684
- AUROC: 0.790440
- F1: 0.166826

## Test timestep evaluation
- Timesteps: 311,992
- Positive timesteps: 5,605
- Timestep prevalence: 1.7965%
- AUPRC: 0.084666
- AUROC: 0.793901
- Sensitivity: 27.07%
- Specificity: 96.45%
- Precision: 12.25%
- F1: 0.168659
- Accuracy: 95.21%
- Confusion matrix: TN=295,520, FP=10,867, FN=4,088, TP=1,517

## Test patient-level evaluation
- Patients: 8,068
- Sepsis-positive patients: 586
- Patient prevalence: 7.263%
- TP: 203; TN: 7,312; FP: 170; FN: 383
- Sensitivity: 34.64%
- Specificity: 97.73%
- Precision: 54.42%
- F1: 0.423358
- Patient AUROC using maximum trajectory probability: 0.810366
- Patient AUPRC using maximum trajectory probability: 0.423421

## Alert timing
Timing is relative to the **first positive SepsisLabel timestep**, not independently verified clinical onset.
- EARLY: 154
- AT_ONSET: 4
- LATE: 45
- MISSED: 383
- FALSE_ALARM: 170
- CLEAN: 7312

Among 154 EARLY patients:
- Mean lead: 58.02 h
- Median: 37.00 h
- P25: 15.25 h
- P75: 85.75 h
- Minimum: 1.00 h
- Maximum: 262.00 h

## Caveats
1. Test predictions had previously been explored during Stage 3.6 screening; Stage 4 uses them only for final evaluation and does not retune the model or threshold.
2. F1/F2/F3 feature-ablation models were screened at 5 epochs, while the earlier F0 family winners used the 15-epoch protocol.
3. This is a comparative methodological study, not a claim of clinical readiness or SOTA.
4. Timestep false positives include alerts before the first positive Challenge label. Those can be classified as EARLY at the patient level and should not be equated with patient-level FALSE_ALARM.
