import streamlit as st
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from segmentation import segmentation, load_nifti

colors = [
    [0, 0, 0, 0],  # Background (fully transparent)
    [0, 0, 1, 1],  # Necrotic/Core (blue)
    [1, 1, 0, 1],  # Edema (yellow)
    [1, 0, 0, 1],  # Enhancing (red)
]

# custom colormap
custom_cmap = ListedColormap(colors)


st.set_page_config(page_title = "Brain Tumor Segmenter", page_icon=":brain:",initial_sidebar_state="collapsed", layout="wide")

# custom css
css = """
<style>
    .block-container{
        padding-top: 2rem;
        padding-bottom: 1rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-size:0.8em;
    }
    div.stButton {
        height:50%;
        width: 50%;
        font-size: 10px;
        paddin: 0.5em 1em;
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# uploads dir
os.makedirs("uploads", exist_ok=True)

@st.cache_resource
def load_segmentation_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Usage in your Streamlit app
model_path = "models/TumorSegmentation.h5"
model = load_segmentation_model(model_path)


def save_uploaded_file(uploaded_file):
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def plot_slice(image, mask=None, slice_idx=0):
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.imshow(image[:, :, slice_idx], cmap="gray")
    if mask is not None:
        ax.imshow(
            mask[:, :, slice_idx], cmap=custom_cmap, alpha=0.5
        )  # Overlay mask with transparency
    ax.axis("off")
    st.pyplot(fig, pad_inches=0, use_container_width=True)


def reset_segmentation_state():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.session_state.mask_img = None
    st.session_state.segmentation_ready = False
    st.session_state.example_selected = None
    st.session_state.flair_file_path = None
    st.session_state.t1_file_path = None
    st.session_state.t2_file_path = None
    st.session_state.t1ce_file_path = None
    if os.path.exists("segmentation.nii"):
        os.remove("segmentation.nii")
    if os.path.exists("uploads"):
        for item in os.listdir("uploads"):
            item_path = os.path.join("uploads", item)
            os.remove(item_path)


def load_example_files(example_name):
    st.session_state.example_selected = example_name
    base_dir = os.path.join("examples", example_name)
    st.session_state.flair_file_path = os.path.join(
        base_dir, f"{example_name}_flair.nii"
    )
    st.session_state.t1_file_path = os.path.join(base_dir, f"{example_name}_t1.nii")
    st.session_state.t2_file_path = os.path.join(base_dir, f"{example_name}_t2.nii")
    st.session_state.t1ce_file_path = os.path.join(base_dir, f"{example_name}_t1ce.nii")


def display_files_with_download():
    if st.session_state.example_selected:
        base_dir = os.path.join("examples", st.session_state.example_selected)

        st.caption(f"Files from : :red[{st.session_state.example_selected}]")

        st.caption(
            f"{st.session_state.example_selected}_flair.nii {st.session_state.example_selected}_t1.nii {st.session_state.example_selected}_t2.nii {st.session_state.example_selected}_t1ce.nii"
        )

        with open(
            os.path.join(base_dir, f"{st.session_state.example_selected}.zip"), "rb"
        ) as file:
            st.download_button(
                label="Download Zipped Files",
                data=file,
                file_name=f"{st.session_state.example_selected}_niifiles.zip",
                mime="application/zip",
            )


def display_tumor_info():
    st.markdown("#### Mapping: ", unsafe_allow_html=True)
    st.markdown(
        " :blue-background[Necrotic Tumor] ",
        unsafe_allow_html=True,
    )
    st.markdown(
        " :orange-background[Peritumoral Edema] ",
        unsafe_allow_html=True,
    )
    st.markdown(
        " :red-background[Enhancing Tumor] ",
        unsafe_allow_html=True,
    )


def quick_start():
    st.markdown("#### Quick Start Guide ", unsafe_allow_html=True)
    st.markdown(
        "- Choose a pre-loaded example or upload your own files (upload all four).",
        unsafe_allow_html=True,
    )
    st.markdown(
        "- Move the slice slider to inspect unsegmented FLAIR brain slices.",
        unsafe_allow_html=True,
    )
    st.markdown(
        "- Press :grey-background[Run Segmentation] to segment the selected volume.",
        unsafe_allow_html=True,
    )


# initialize session states

if "mask_img" not in st.session_state:
    st.session_state.mask_img = None
if "segmentation_ready" not in st.session_state:
    st.session_state.segmentation_ready = False
if "example_selected" not in st.session_state:
    st.session_state.example_selected = None
if "flair_file_path" not in st.session_state:
    st.session_state.flair_file_path = None
if "t1_file_path" not in st.session_state:
    st.session_state.t1_file_path = None
if "t2_file_path" not in st.session_state:
    st.session_state.t2_file_path = None
if "t1ce_file_path" not in st.session_state:
    st.session_state.t1ce_file_path = None

# Streamlit app
col_side, col_main = st.columns([3, 5], gap="large")
with col_side:
    st.markdown(
        "<h3 style='text-align: left;'>&#129504<u>Brain Tumor Segmenter</u> </h3>",
        unsafe_allow_html=True,
    )
    st.markdown("<p style = 'text-align : left; font-size: 1em; color: gray;'>Segment upto 3 types of tumors from brain MRI. For source code, model and training process check <a href = 'https://github.com/Swastik-Aryal/Brain-Tumor-Segmenter'>GitHub</a></p>",unsafe_allow_html=True)

    if st.button("Reset All", key="reset", help="Reset the app"):
        reset_segmentation_state()

    # examples
    
    st.subheader("Use Examples", divider="red")
    st.markdown("<p style = 'text-align : left; font-size: 1em; color: gray;'>* make sure there is no data uploaded.</p>",unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)

    if col1.button("patient001", key="ex1", help="Use Example 1"):
        reset_segmentation_state()
        load_example_files("patient001")
    if col2.button("patient002", key="ex2", help="Use Example 2"):
        reset_segmentation_state()
        load_example_files("patient002")
    if col3.button("patient003", key="ex3", help="Use Example 3"):
        reset_segmentation_state()
        load_example_files("patient003")

    display_files_with_download()

    st.subheader(" :red[OR] ")

    # file uploaders
    st.subheader("Upload MRI Images", divider="red")
    flair_uploaded = st.file_uploader(
        "Upload flair.nii/nii.gz file", type=["nii", "nii.gz"]
    )
    t1_uploaded = st.file_uploader("Upload t1.nii/nii.gz file", type=["nii", "nii.gz"])
    t2_uploaded = st.file_uploader("Upload t2.nii/nii.gz file", type=["nii", "nii.gz"])
    t1ce_uploaded = st.file_uploader(
        "Upload t1ce.nii/nii.gz file", type=["nii", "nii.gz"]
    )


with col_main:
    # if a new file is uploaded, reset the example state
    if flair_uploaded or t1_uploaded or t2_uploaded or t1ce_uploaded:
        if st.session_state.example_selected:
            reset_segmentation_state()
        st.session_state.example_selected = None

    # determine whether to use example files or uploaded files
    if st.session_state.example_selected:
        flair_file = st.session_state.flair_file_path
        t1_file = st.session_state.t1_file_path
        t2_file = st.session_state.t2_file_path
        t1ce_file = st.session_state.t1ce_file_path
    else:
        flair_file = save_uploaded_file(flair_uploaded) if flair_uploaded else None
        t1_file = save_uploaded_file(t1_uploaded) if t1_uploaded else None
        t2_file = save_uploaded_file(t2_uploaded) if t2_uploaded else None
        t1ce_file = save_uploaded_file(t1ce_uploaded) if t1ce_uploaded else None

    # proceed if all files are provided (either uploaded or from an example)
    if flair_file and t1_file and t2_file and t1ce_file:
        flair_image = load_nifti(flair_file)

        # Slider to select a slice
        slider_custom = "Select Brain Slice across the axial plane. _Showing_ :red[User Uploaded Data]"
        slider_ex = f"Select Brain Slice across the axial plane. _Showing_ :red[{st.session_state.example_selected}]"
        slice_idx = st.slider(
            slider_ex if st.session_state.example_selected else slider_custom,
            0,
            flair_image.shape[2] - 1,
            (flair_image.shape[2] - 1) // 2,
        )

        run_segmentation = st.button(
            "Run Segmentation",
            key="run_segmentation",
            help="Perform Segmentation on current volume",
        )

        if run_segmentation:
            with st.spinner("Wait for segmentation to complete..."):
                # Perform segmentation
                mask_img = segmentation(flair_file, t1_file, t2_file, t1ce_file, model)

            # Store the results in session state
            st.session_state.mask_img = mask_img
            st.session_state.segmentation_ready = True

            # Save the segmentation to a NIfTI file
            segmentation_nii = nib.Nifti1Image(mask_img, np.eye(4))
            nib.save(segmentation_nii, "segmentation.nii")

        if st.session_state.mask_img is not None:
            col_img, col_txt = st.columns([3, 2])
            with col_img:
                plot_slice(flair_image, st.session_state.mask_img, slice_idx)
            with col_txt:
                display_tumor_info()

        else:
            with st.columns([3, 2])[0]:
                plot_slice(flair_image, slice_idx=slice_idx)
        if st.session_state.segmentation_ready:
            st.download_button(
                label="Download Segmentation",
                data=open("segmentation.nii", "rb"),
                file_name="segmentation.nii",
                mime="application/gzip",
                key="download_segmentation",
            )

    else:
        quick_start()
        st.warning(
            "If selecting an example, make sure you have removed all files from input fields to avoid complications."
        )
        st.warning(
            "For user-uploaded data, the program works automatically when all four files have been uploaded."
        )
