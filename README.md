# PiE: Prior Images Guided Deep AutoEncoder Model for Dual Camera Spectral Compressive Imaging

- **Abstract**

*Compressive Spectral Imaging (CSI) techniques have attracted considerable attention among researchers for their ability to simultaneously capture spatial and spectral information using low-cost, compact optical components. A prominent example of CSI techniques is the Dual-Camera Coded Aperture Snapshot Spectral Imaging (DC-CASSI), which involves reconstructing hyperspectral images from CASSI measurements and uncoded panchromatic or RGB images. Despite its significance, the reconstruction process in DC-CASSI is challenging. Conventional DC-CASSI techniques rely on different models to explore the similarity between uncoded images and hyperspectral images. Nevertheless, two main issues persist: i) the effective utilization of spatial information from RGB images to guide the reconstruction process, and ii) the enhancement of spectral consistency of recovered images when using panchromatic/RGB images, which inherently lack precise spectral information. To address these challenges, we propose a novel Prior images guided generative autoEncoder (PiE) model. The PiE model leverages RGB images as prior information to enhance spatial details and designs a generative model to improve spectral quality. Notably. the generative model is optimized in a self-supervised manner. Comprehensive experimental results demonstrate that the proposed PiE method outperforms existing techniques, achieving state-of-the-art performance.*

---

- **Dataset preparation**

Download the testing dataset from our [previous project](https://github.com/YurongChen1998/Prior-Image-Guided-Snapshot-Spectral-Compressive-Imaging).

Put the data into *Dataset* folder.

---

- **Testing results**

Running *main_KAIST.py* or *main_CAVE.py* for reconstructing one scene.

---
If you have any problem, please contact me: chenyurong1998 at outlook.com
