# Negative Spotlight Overlap Penalty

The current spotlight overlap loss evaluates candidates within twice the light range (`d < 2r`) and applies the paper's weighted softmin to `r - d` and the angular boundary distance. For candidates outside the actual range (`r < d < 2r`), `r - d` is negative, so the resulting penalty can also be negative even when the light contributes zero irradiance.

These negative contributions are amplified by the overlap-count factor and can dominate the total objective. In the Sponza run, overlap decreased from approximately -20,657 to -55,950. Reconstruction error reached 45.78 at step 150, then increased to 73.49 at step 300 while the total objective continued decreasing. A lower total therefore does not necessarily indicate better reconstruction or less actual overlap.

The paper explicitly uses doubled influence ranges and this signed softmin, so this is not established as a transcription error. The issue is diagnosed but not fixed. Candidate counting and boundary penalties need review; proposed nonnegative formulations remain experimental alternatives.

Evidence: `tmp/validation/negative_overlap/report.json` and `data/experiment_sponza.json`.
