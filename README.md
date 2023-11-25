# Tumor Segmentation Project 🧬🖥️

## Introduction
This Tumor Segmentation project, was created for DM i AI 2023 - Denmark's AI competition for students. It focuses on segmenting tumors from whole-body MIP-PET images.
An `Attention U-Net` model will for each pixel in the image, the model predicts whether it belongs to a tumor or a healthy area.

The project included a lot of learning and a model I never used before so it's by far perfect and although I couldn't race against the clock to test my model in the competition, I think it does a pretty decent job. Achving a Dice-Score of `0.86` on the validation set with only cancer cases. Post-competition, I added a Streamlit app that takes this project from competition entry to a small app to play around with.

## Project Structure 📂

This project has a simplestructure:

- `experimentation`: Contains Jupyter notebooks like `experimenting.ipynb` where the model is trained and validated.
- `src`:
  - `app`: Houses the Streamlit application (`app.py`)
  - `model`: Includes the `attention_unet.py` (the model architecture and infrens method) and the trained model file `best_metric_model_segmentation2d_dict.pth`.
  - `tests`: For any future testing and validation scripts.
- Root Directory: Contains essential files like `.gitignore`, `README.md` and configuration files (`poetry.lock`, `pyproject.toml`) for dependency management.

## Approach


- **Research**: Explored academic resources to find effective segmentation methods. This led to the MONAI library and the Attention U-Net model.

- **Refinement**: Ihe implementation of the DiceFocal loss function, inspired by a [study on Whole-Body MIP-PET Imaging](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.16438). This significantly improved model accuracy, especially in segmenting smaller tumor sections.

In essence, this project was a blend of research, practical implementation, and a lot of trail an error.

## Built With

- [MONAI](https://monai.io/): A PyTorch-based framework for deep learning in healthcare imaging.
- [PyTorch](https://pytorch.org/): An open source machine learning framework.

## Authors

- @MagnusS0 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
