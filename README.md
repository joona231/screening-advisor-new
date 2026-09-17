# Screening Advisor — AI-PQI

Artificial Intelligence-assisted Print Quality Intelligence (AI-PQI).

Purpose:
A data-driven decision-support prototype for selecting the paper substrate and
screening method expected to minimize colour reproduction error (DeltaE00).

Experimental dataset:
- 4 substrates
- 6 screening methods
- 400 CMYK patches
- 9,600 measured observations
- D50 illuminant
- 10° observer
- FOGRA39 colour-managed reference condition

Model:
Weighted 12-nearest-neighbour regression in normalized CMYK space.
The application deliberately avoids scikit-learn and SciPy so it runs in
restricted Windows environments where compiled SciPy DLLs may be blocked.

Validation:
Five-fold grouped cross-validation by Patch ID. The same patch is never
simultaneously used for training and testing.

Colour management:
- CMYK -> Lab: bundled FOGRA39 ICC profile via LittleCMS/Pillow.
- Optional RGB -> CMYK: sRGB -> FOGRA39 ICC.
- Experimental Target Lab values remain separate from ICC-derived reference Lab.

Run:
    pip install -r requirements.txt
    streamlit run app.py
