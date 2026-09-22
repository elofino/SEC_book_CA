
# Reversible chronoamperometry + UV–Vis SEC Streamlit app

This app simulates a simple reversible one-electron transfer

    O + e- <=> R

at a planar electrode following a potential step from Ei to Ef.

## Model
- 1D planar diffusion.
- Equal diffusion coefficient for O and R.
- Nernstian interfacial equilibrium (fully reversible electron transfer).
- Finite spatial domain.
- Zero-flux (Neumann) condition at the outer boundary.
- Initial composition can be either:
  - Nernst equilibrium at Ei, or
  - user-defined fraction of R.
- No homogeneous chemistry.
- No migration or convection.
- No double-layer charging or uncompensated resistance.

The app calculates:
1. chronoamperometric current;
2. concentration profiles;
3. UV–Vis absorbance / delta-absorbance.

Optical options:
- normal transmission;
- near-normal reflection (idealised double pass);
- parallel mode using a finite sampling layer located between x0 and x0 + beam thickness.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Sign convention
Cathodic reduction current is plotted as negative.

## Important numerical point
Because the outer boundary is zero flux rather than fixed concentration, the transient
crosses over from semi-infinite-like diffusion to finite-domain behaviour when the diffusion
length becomes comparable with the selected domain length.
