# Tumor Segmentation Project 🧬🖥️

## Introduction
This Tumor Segmentation project, was created for [DM i AI 2023](https://github.com/amboltio/DM-i-AI-2023/tree/main/tumor-segmentation) - Denmark's AI competition for students. It focuses on segmenting tumors from whole-body MIP-PET images.
An `Attention U-Net` model will for each pixel in the image, predict whether it belongs to a tumor or a healthy area.

The project included a lot of learning and a model I never used before so it's by far perfect. I couldn't race against the clock to test my model in the competition, but it achving a Dice-Score of `0.86` on the validation set with only cancer cases. On a proper test set I assume the score would be lower. Post-competition, I added a small Streamlit app to test how the model could work in a real setting.

## Approach


- **Research**: The choice of the Attention U-Net model was driven by researching diffrent segmentation methods in medical imaging. I quickly found that Attention U-Net's where one of the top performers on benchmarks such as PapersWithCode's [Image Segmentation](https://paperswithcode.com/task/image-segmentation) and [Tumor Segmentation](https://paperswithcode.com/task/tumor-segmentation) lists. This architecture, as detailed in the paper [Attention U-Net: Learning Where to Look for the Pancreas](https://arxiv.org/abs/1804.03999), is escpesilly good at isolating critical features within medical images, and can be used to replace external organ localization models. This is crucial in MIP-PET, where organs like the brain, liver, bladder, and kidneys often exhibit high sugar uptake, resembling tumor characteristics. This is done through self-Attention Gates (AGs). In essence, these AGs selectively focus on relevant spatial regions, filtering out background noise and irrelevant features. So through training the model will e.g. start focusing away from organ areas and focuse it's attention on more commen tumor regions. 

- **Refinement**: Implementing the DiceFocal loss function was inspired by a [study on Whole-Body MIP-PET Imaging](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.16438), which highlighted the challenges in detecting smaller tumors. This adjustment significantly enhanced model precision, as it focuses the model more on the hard, misclassified examples and less on the easy, well-classified examples (like the large areas without tumors). This is refelcted in the lambda values set to 1 for Dice and 10 for Focal, adopted from the paper.
  
In essence, this project was a blend of research, practical implementation, and a lot of trail an error.

## Project Structure 📂

This project has a simplestructure:

- `experimentation`: Contains Jupyter notebooks like `experimenting.ipynb` where the model is trained and validated.
- `src`:
  - `app`: Houses the Streamlit application (`app.py`)
  - `model`: Includes the `attention_unet.py` (the model architecture and infrens method) and the trained model file `best_metric_model_segmentation2d_dict.pth`.
  - `tests`: For any future testing and validation scripts.
- Root Directory: Contains essential files like `.gitignore`, `README.md` and configuration files (`poetry.lock`, `pyproject.toml`) for dependency management.

## Built With

- [MONAI](https://monai.io/): A PyTorch-based framework for deep learning in healthcare imaging.
- [PyTorch](https://pytorch.org/): An open source machine learning framework.

  
## Sources
- Cardoso, M. J., Li, W., Brown, R., Ma, N., Kerfoot, E., Wang, Y., ... & Feng, A. (2022). Monai: An open-source framework for deep learning in healthcare. arXiv preprint arXiv:2211.02701.
- He J, Zhang Y, Chung M, et al. Whole-body tumor segmentation from PET/CT images using a two-stage cascaded neural network with camouflaged object detection mechanisms. Med Phys. 2023; 50: 6151–6162. https://doi.org/10.1002/mp.16438
- Oktay, O., Schlemper, J., Folgoc, L. L., Lee, M., Heinrich, M., Misawa, K., ... & Rueckert, D. (2018). Attention u-net: Learning where to look for the pancreas. arXiv preprint arXiv:1804.03999.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
