import cv2

import numpy as np
import streamlit as st

from utils import TablePipe
from .config_reader import config


reference = cv2.imread(config.signature_best_sheet)
uploaded_file = st.file_uploader("Upload Files", type=['png', 'jpeg', 'jpg'])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    pipe = TablePipe()
    res_image = pipe.run(image, reference)
    st.image(res_image)
