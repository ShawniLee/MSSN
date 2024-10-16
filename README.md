# Multi-Scale Time Series Segmentation Network Based on Eddy Current Testing for Detecting Surface Metal Defects

[//]: # ([![Paper Title]&#40;link-to-paper&#41;]&#40;https://link-to-your-paper&#41;  )

[//]: # (_Short description of the paper &#40;e.g., Deep Learning-based Industrial Anomaly Detection&#41;_)

## Overview
The overview of the paper will be made public after publication.

## Project Structure

```
C:.
│  .gitignore
│  mackey_glass_npy_test.py     # A test script for specific experiments
│  train_MG.py                  # Script for training the model
├─best_test                     # The model parameter of the best test runs
├─layers                        # Custom neural network layers or model components.
├─mg_dataset                    # The Mackey-Glass dataset
├─mg_metrics_csv                # Evaluation metrics generated during testing
├─models                        # Comparison method

```

## Requirements

The project was developed and tested in the following environment. Please make sure you have the following dependencies installed:

- Python version: `Python >= 3.9`
- Required packages: Install all dependencies by running the following command:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

### 1. Data Preprocessing

Run the following command to preprocess the data:

```bash
python scripts/preprocess_data.py --input_dir data/raw --output_dir data/processed
```

### 2. Model Training

To train the model, use the following command:

```bash
python train_MG.py
```

The trained model will be saved in the `best_test/` folder.


### 3. Model Testing

To test the model, run:

```bash
python mackey_glass_npy_test.py```

Test results will be printed in the terminal and saved in the `mg_metrics_csv/` folder.
## Citation

If you find our code or data useful in your research, please cite our paper as follows:

```
@article{your-paper-reference,
  title={Paper Title},
  author={List of Authors},
  journal={Journal Name},
  year={2023}
}
```

## License

This project is licensed under the [MIT License](LICENSE). For more details, please refer to the LICENSE file.

## Contact

For questions, please contact `your-email@example.com` or open an [issue](https://github.com/your-repo/issues).