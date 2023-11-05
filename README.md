# Multimodal Deep Learning Framework for Biomarker Prediction

## Description
This project introduces a multimodal deep learning framework designed to predict two types of biomarkers: stromal tumor-infiltrating lymphocytes (sTILs) in breast cancer and Microsatellite Instability (MSI) in colorectal cancer. Utilizing pretrained RetCCL [1] for image feature extraction from H&E-stained whole-slide images (WSIs) and BioBERT [2] for text feature extraction from diagnostic reports, our framework achieves state-of-the-art performance in predicting these biomarkers. The data for this study is sourced from The Cancer Genome Atlas (TCGA) [3]. Our results demonstrate the superiority of our multimodal approach over traditional image-only methods, achieving a correlation of 0.71 ± 0.03 for sTILs prediction and an AUROC of 0.76 ± 0.08 for MSI classification.

The implementation is done in Python 3 on a single A100 GPU, leveraging PyTorch, PyTorch Lightning, and HuggingFace libraries. Note that CUDA/GPU and Python >=3.9 are required for compatibility.

## Code Structure
```
.
├── .gitignore
├── data_utils.ipynb
├── multimodal.ipynb
├── requirements.txt
└── src
    ├── extract_wsi_feats.py
    ├── models.py
    ├── process_reports.py
    ├── train.py
    └── utils.py
```

- `src/extract_wsi_feats.py`: Functions to extract features from WSIs using RetCCL.
- `src/models.py`: Defines the main multimodal regressor and classifier classes.
- `src/process_reports.py`: Functions to process reports including OCR, distillation, summarization with GPT-3, and feature extraction with BioBERT.
- `src/train.py`: Functions for model training and performing k-fold cross-validation.
- `src/utils.py`: Contains the multimodal dataset class, data loader creation functions, and various helper functions.
- `data_utils.ipynb`: Jupyter notebook for data downloading and processing.
- `multimodal.ipynb`: Jupyter notebook for feature extraction, model training, and evaluation.
- `requirements.txt`: List of required Python packages.

## Usage

### Install env & dependencies
Create and activate a new environment:
   ```bash
   conda create -n multimodal python==3.11
   conda activate multimodal
   pip install -r requirements.txt
   ```

### Running the experiments

1. Download the TCGA data for BRCA and CRC subtypes from the [GDC Data Portal](https://portal.gdc.cancer.gov/). Note that approximately 1.5 TB of storage is required for TCGA-BRCA and 800 GB for TCGA-COAD.

2. Install libvips for your system as per the instructions [here](https://www.libvips.org/install.html).

3. Create a `.env` file in the root directory with your `HF_API_KEY` (for HuggingFace Transformers) and `OPENAI_API_KEY` (for GPT-3).

4. Run the cells in `data_utils.ipynb` to preprocess the data.

5. Execute the cells in `multimodal.ipynb` to extract features, train, and evaluate the models.

6. To view runtime logs, execute:
   ```bash
   tensorboard --logdir lightning_logs
   ```

## References
1. Wang, X., Du, Y., Yang, S., Zhang, J., Wang, M., Zhang, J., ... & Han, X. (2023). RetCCL: clustering-guided contrastive learning for whole-slide image retrieval. Medical image analysis, 83, 102645.

2. Lee, J., Yoon, W., Kim, S., Kim, D., Kim, S., So, C. H., & Kang, J. (2020). BioBERT: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics, 36(4), 1234-1240.

3. Cancer Genome Atlas Network. (2012). Comprehensive molecular characterization of human colon and rectal cancer. Nature, 487(7407), 330.