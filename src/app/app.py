import streamlit as st
import os
import tempfile
from PIL import Image
from io import BytesIO
from model.attention_unet import AttentionUnetModel


st.set_page_config(layout="wide", page_title="Tumor Segmentation using Attention U-Net")

st.title("Tumor Segmentation using Attention U-Net")
st.markdown("""
    Welcome to the tumor segmentation app! In this application, you can upload 
    whole-body MIP-PET images of patients with cancer, and the Attention U-Net model 
    will attempt to segment tumorous areas. For each pixel in the image, the model 
    predicts whether it belongs to a tumor or a healthy area.
    
    **Instructions:**
    - Upload a Whole-body MIP-PET image of a patient with cancer.
    - Review the model's segmentation output on the right.
            
    **Please note:** This tool is for demonstration and educational purposes only and should not be used for medical 
    diagnosis or treatment.
""")
# Set up the sidebar with instructions and file uploader
st.sidebar.header("Upload and Process Image :gear:")
st.sidebar.write("Please upload a MIP-PET image in PNGformat. The maximum file size is 5MB.")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Assuming the AttentionUnetModel class is defined as per your previous code
model = AttentionUnetModel(model_path="src/model/best_metric_model_segmentation2d_dict.pth")

def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def process_image(upload):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
        tmpfile.write(upload.getvalue())
        tmpfile_path = tmpfile.name
    try:
        image = Image.open(upload)
        col1.image(image)

        # Process the image with the model
        processed = model.infer(tmpfile_path)
        col2.image(processed)
        st.sidebar.markdown("\n")
        st.sidebar.download_button("Download segmented image", convert_image(processed), "segmented.png", "image/png")

    finally:
        # Clean up the temporary file
        os.remove(tmpfile_path)

col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

# Check if an image has been uploaded
if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        # Process and display the images in the respective columns
        col1.subheader("Original Image :camera:")
        col2.subheader("Segmented Image :mag:")
        process_image(upload=my_upload)
else:
    # Display a warning message if no image has been uploaded yet
    st.sidebar.warning("Awaiting image upload...")

# Placeholder for additional app information or instructions
st.info("Upload an image to begin analysis. The resulting segmentation will appear on the right.")
