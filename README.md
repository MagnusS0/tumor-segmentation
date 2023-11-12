# Tumor Segmentation Project with MONAI

This project aims to segment tumors from Whole-body MIP-PET images using U-Net, V-Net, and potentially LFBNet models. The project uses MONAI for preprocessing and data augmentation. The performance of the models is evaluated using the Sørensen-Dice coefficient.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.8 or higher
- Poetry for Python dependency management

### Installing

1. Clone the repository to your local machine.

```bash
git clone https://github.com/yourusername/tumor-segmentation.git
```

2. Navigate to the project directory.

```bash
cd tumor-segmentation
```

3. Install the project dependencies using Poetry.

```bash
poetry install
```

## Running the Project

1. Activate the Poetry environment.

```bash
poetry shell
```

2. Run the main script.

```bash
python src/main.py
```

## Project Structure

The project has the following structure:

- `src/main.py`: Main script that runs the entire pipeline.
- `src/models/`: Contains the implementation of the U-Net, V-Net, and LFBNet models.
- `src/preprocessing/monai_preprocess.py`: Contains the preprocessing steps using MONAI.
- `src/utils/utils.py`: Contains utility functions used throughout the project.
- `src/tests/test_models.py`: Contains tests for the models.
- `data/mip_pet_images`: Directory containing the MIP-PET images used for training and testing the models.
- `poetry.lock` and `pyproject.toml`: Used by Poetry for dependency management.

## Testing

To run the tests for the models, navigate to the `src/tests` directory and run the `test_models.py` script.

```bash
cd src/tests
python test_models.py
```

## Built With

- [MONAI](https://monai.io/): A PyTorch-based framework for deep learning in healthcare imaging.
- [PyTorch](https://pytorch.org/): An open source machine learning framework.

## Authors

- Your Name

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.