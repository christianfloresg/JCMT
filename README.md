# JCMT
Here are the codes necessary to process raw data from JCMT HARPS in jiggle mode and then obtain concentration factors

## Gas-centered concentration and envelope classification

The legacy `concentration_factors.py` is retained for reproducibility.  Its
Gaussian result is strongly tied to the SIMBAD/IR position: the peak is found
inside the YSO-centered beam and the Gaussian center can move only about one
arcsecond.

For the separated morphology/association analysis, run:

```bash
python gas_centered_concentration.py \
  --out text_files/concentrations_gascentered_mc.csv --nmc 100

python classify_and_plot_envelopes.py \
  --measurements text_files/concentrations_gascentered_mc.csv \
  --outdir envelope_classification_outputs
```

The revised workflow reports three distinct observables:

- gas-centered concentration `C_gas`;
- HCO+ integrated intensity in the beam centered on the IR YSO;
- angular offset between the IR YSO and the selected molecular-line peak.

The primary concentration uses the unmodified moment-zero data inside the
fitted FWHM.  A background-subtracted concentration is also reported as a
sensitivity test, but should not be substituted silently because the published
0.6 threshold was not calibrated for every possible background model.

Classification rules are applied hierarchically: a significant off-source
component beyond 20 arcsec with no significant on-source peak is `Confused`;
otherwise every embedded criterion must pass; remaining objects are
`Non-embedded` with measurement and uncertainty flags retained.
