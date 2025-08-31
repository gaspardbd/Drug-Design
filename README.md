# Drug Design with Diffusion Models

This project was conducted at **École des Ponts ParisTech**, under the supervision of **Paraskevi Gkeka, Gabriel Stoltz, Tony Levièvre, and Régis Santet**.  
The goal was to explore how **diffusion models** can be applied to **drug design**, starting from fundamentals in image generation to protein-ligand docking.

---

## Project Overview

The project is divided into two main parts:

### 1. Diffusion Models for Images
- Implemented a simple **Denoising Diffusion Probabilistic Model (DDPM)** on the **Fashion-MNIST** dataset.  
- The implementation was inspired by an open-source notebook from the **CNRS**.  
- Code is available in the [`image_generation`](./image_generation) folder.

### 2. DiffDock for Drug Design
- Studied the **DiffDock** model from **CSAIL, MIT** (Gabriele Corso, Hannes Stärk, Bowen Jing, Regina Barzilay, Tommi Jaakkola).  
- Adapted a Colab notebook ([original here](https://colab.research.google.com/drive/1CTtUGg05-2MtlWmfJhqzLTtkDDaxCDOQ)) to simulate **ligand docking on the BCR-ABL protein complex**.  
- Explored docking procedures, analyzed results, and visualized interactions.  
- Performed a **comparative study between DiffDock and GNINA**, focusing on **RMSD (Root Mean Square Deviation)** metrics to evaluate prediction accuracy.  
- Code and experiments are available in the [`diff_dock`](./diff_dock) folder.


## References

- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.  
- Corso, G., Stärk, H., Jing, B., Barzilay, R., & Jaakkola, T. (2023). *DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking.* CSAIL, MIT.
