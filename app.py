import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import urllib.request


# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷"
)


st.title("😷 Face Mask Detection using CNN")

st.write(
    "Upload a face image and detect whether the person is wearing a mask or not."
)


# -------------------------------
# Classes
# -------------------------------

classes = [
    "with_mask",
    "without_mask"
]


# -------------------------------
# CNN Architecture
# Same as training
# -------------------------------

class FaceMaskCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128*28*28,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                512,
                2
            )

        )


    def forward(self,x):

        x = self.features(x)

        x = self.classifier(x)

        return x



# -------------------------------
# Model Download
# -------------------------------

MODEL_PATH = "best_face_mask_model.pth"


# Paste your direct download link here

MODEL_URL = "PASTE_MODEL_LINK_HERE"



def download_model():

    if not os.path.exists(MODEL_PATH):

        with st.spinner("Downloading model..."):

            urllib.request.urlretrieve(
                MODEL_URL,
                MODEL_PATH
            )



# -------------------------------
# Load Model
# -------------------------------

@st.cache_resource
def load_model():

    download_model()

    model = FaceMaskCNN()


    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )


    model.load_state_dict(
        checkpoint
    )


    model.eval()


    return model



model = load_model()



# -------------------------------
# Image Transform
# -------------------------------

transform = transforms.Compose([

    transforms.Resize(
        (224,224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]

    )

])



# -------------------------------
# Upload Image
# -------------------------------

uploaded_file = st.file_uploader(

    "Upload Face Image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]

)



if uploaded_file:


    image = Image.open(
        uploaded_file
    ).convert("RGB")


    st.image(

        image,

        caption="Uploaded Image",

        use_container_width=True

    )


    img = transform(image)


    img = img.unsqueeze(0)



    with torch.no_grad():


        output = model(img)


        probability = torch.softmax(

            output,

            dim=1

        )


        confidence, prediction = torch.max(

            probability,

            1

        )



    result = classes[
        prediction.item()
    ]


    confidence = confidence.item()*100



    if result == "with_mask":

        st.success(

            f"😷 Mask Detected\n\nConfidence: {confidence:.2f}%"

        )


    else:

        st.error(

            f"😐 No Mask Detected\n\nConfidence: {confidence:.2f}%"

        )