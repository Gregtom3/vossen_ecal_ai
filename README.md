[![CI](https://github.com/Gregtom3/vossen_ecal_ai/actions/workflows/ci.yml/badge.svg)](https://github.com/Gregtom3/vossen_ecal_ai/actions/workflows/ci.yml)
[![Run Examples](https://github.com/Gregtom3/vossen_ecal_ai/actions/workflows/run-examples.yml/badge.svg)](https://github.com/Gregtom3/vossen_ecal_ai/actions/workflows/run-examples.yml)

! This repository is a work in progress ! Most of the current work can be found in `notebooks/`.

# AI-assisted calorimeter clustering at CLAS12

---

This repository documents two separate AI projects involving the CLAS12 detector.

  - Gradient Boosted Decision Trees for removing false photons in the event particle list
  - A GravNet + Transformer encoder network using object condensation to perform clustering of ECal hits

Reproducible code in the form of Google Colab notebooks are provided in `notebooks/`. These notebooks train models very similar to those used in the published analyses. For the GBT model, see notebook `notebooks/nb01...`. Notebooks 2-5 introduce some basic datasets used for the AI clustering and are intended to be run one after another. 


