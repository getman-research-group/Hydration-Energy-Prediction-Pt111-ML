# Prediction of hydration energies of adsorbates at Pt(111) and liquid water interfaces using machine learning

![Workflow Diagram](https://github.com/getman-research-group/Hydration-Energy-Prediction-Pt111-ML/blob/main/figure_1.png)  
**Figure 1.** An overview of the workflow for this work. The MD trajectory and configurations are used to generate features for ML, while single-point DFT calculations provide the interaction energies as labels for ML training.


---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Dependencies](#-dependencies)
- [Usage Guide](#-usage-guide)

---

## 🔍 Project Overview
This repository contains the computational scripts supporting the research article:  
**"Prediction of Hydration Energies of Adsorbates at Pt(111) and Liquid Water Interfaces Using Machine Learning"**  
*(Submitted to Journal of Chemical Physics)*  

The code enables:  
✅ Molecular fingerprint generation for adsorbate species  
✅ Configuration-based descriptor calculations  
✅ Trajectory-based feature extraction from MD simulations  
✅ ML model training for hydration energy prediction  

---

## 🗂 Repository Structure

├── database/                          # Precomputed datasets for ML models
│   ├── adsorbate_fingerprints/        # Molecular fingerprints of adsorbates (precalculated)
│   └── label_data/                    # Target values for ML training/test
│
├── md_simulations/                    # Molecular dynamics (MD) output files for calculating features
│   └── xyz_files/                     # Optimized adsorbate configurations on Pt(111)
│
├── python_scripts_adsorbate/          # Adsorbate fingerprint calculation scripts
│   └── core/                          # Core functional scripts
│
├── python_scripts_config/             # Configuration model descriptors
│   └── core/                          # Core scripts
│
├── python_scripts_traj/               # Trajectory model descriptors
│   └── core/                          # Core scripts
│
└── README.md                          # This documentation
