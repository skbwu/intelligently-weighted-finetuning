# Intelligently Reweighting Multiple Reference Models for Direct Preference Optimization of LLMs.

This repository accompanies the paper "Intelligently Reweighting Multiple Reference Models for Direct Preference Optimization of LLMs," written by Skyler Wu and Aymen Echarghaoui and submitted as a final course project to Stanford University's [CS 329H: Machine Learning from Human Preferences](https://web.stanford.edu/class/cs329h/) course, taught by Professor Sanmi Koyejo and Dr. Andy Haupt.

**Environment Setup and Dependencies + Expected Runtime and Computational Requirements:** all experiments were run on Google Colab Pro+ using single NVIDIA A100 High-RAM instances with 80 GB of GPU memory each. The only additional packages the Colab user needs to install can be done via the following: `! pip install bitsandbytes==0.46.0 accelerate==1.7.0`. If the reader is 

We consider these settings to be the minimum requirements for running our experiments as even with these A100 instances, significant engineering (see our paper Appendix) was necessary to fit experiments in memory and with reasonable runtime. All experiments for `UltraFeedback` take 1.5 to 2 hours per seeded trial, while all experiments for `SafeRLHF` take 0.5 to 0.75 hours per seeded trial.

**Reproducing All Results:** Which scripts produce which results/figures in the paper? Gradient-clipping on Online 1 (UltraFeedback) and it still went boom.

**Repository Structure and File Organization:**

**Links to Any Required Datasets or Instructions for Data Generation:**

