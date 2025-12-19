Update README with dataset description
# Weakly Supervised Semantic Segmentation of Hydrothermal Alteration Minerals

This repository contains the dataset and supporting materials used in the study:

**Weakly Supervised Semantic Segmentation of Hydrothermal Alteration Minerals in Thin Sections Using a Hybrid CNN–CRF Framework**

The project addresses the challenge of pixel-wise mineral mapping in petrographic thin sections by proposing a weakly supervised multi-class semantic segmentation framework that relies solely on image-level labels, significantly reducing the need for costly manual pixel-level annotations.

---

## Dataset Overview

The dataset consists of high-resolution petrographic thin-section images acquired under cross-polarized light (XPL) conditions. It represents five major hydrothermal alteration classes commonly encountered in economic geology:

- Sericitization  
- Silicification  
- Carbonatization  
- Chloritization  
- Epidotization  

The images were collected from hydrothermal alteration zones associated with porphyry Cu–Au and vein/skarn-type Pb–Zn mineral systems. Thin sections were prepared using standard petrographic procedures to ensure optical consistency, and imaging conditions were kept constant across all samples.

The final dataset contains **5,000 high-resolution microscopy images**, equally distributed across the five alteration classes.

---

## Intended Use

This repository is intended for research and educational purposes in:

- Weakly supervised semantic segmentation  
- Deep learning–based microscopy image analysis  
- Automated petrographic and mineralogical interpretation  
- Geoscientific image analysis and mineral exploration  

The dataset was originally compiled for image-level classification tasks and has been **extended with additional images to support multi-class semantic segmentation experiments**.

---

## Methodological Context

In the associated study, the dataset is used to evaluate a hybrid **CNN–CRF framework**, which integrates:

- EfficientNet-B4 for feature extraction and localization (Grad-CAM),
- Adaptive pseudo-mask generation refined with Dense Conditional Random Fields (DenseCRF),
- U-Net for multi-class semantic segmentation.

This pipeline enables the generation of pixel-wise mineral maps using only image-level supervision.

---

## Notes and Limitations

- The dataset does not include manually annotated pixel-level ground truth masks.
- Images are intended for algorithm development and benchmarking rather than absolute mineral quantification.
- Geological domain knowledge is recommended for interpretation of the results.

---

## Citation

If you use this dataset in your research, please cite the corresponding article:

Ünsal, E. (Year). *Weakly Supervised Semantic Segmentation of Hydrothermal Alteration Minerals in Thin Sections Using a Hybrid CNN–CRF Framework.*

---

## Contact

**Emre Ünsal**  
Department of Software Engineering  
Sivas Cumhuriyet University, Türkiye  
📧 eunsal@cumhuriyet.edu.tr
