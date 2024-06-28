import cv2
import numpy as np
import streamlit as st
from camera_input_live import camera_input_live
from datetime import datetime
import os

"# Streamlit camera input live Demo"
"## Try holding a qr code in front of your webcam"

image = camera_input_live()

if image is not None:

    if os.path.exists('t_file.txt'):
        with open('t_file.txt','r') as f:
            t_old = float(f.readlines()[0])
        t_new = datetime.now()
        t_new = t_new.timestamp()
        diff_t = t_new - t_old
        st.write(f'Time difference = {diff_t:.2f}')
    
    # bytes_data = image.getvalue()
    # cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    # down_image = cv2_img[::5]
    # down_image = down_image[:,::5]
    # st.image(down_image)
    

    t_old = datetime.utcnow()
    st.write(f'Image captured {t_old}')
    with open('t_file.txt','w') as f:
        f.write(str(t_old.timestamp()))


    # detector = cv2.QRCodeDetector()

    # data, bbox, straight_qrcode = detector.detectAndDecode(cv2_img)

    # if data:
    #     st.write("# Found QR code")
    #     st.write(data)
    #     with st.expander("Show details"):
    #         st.write("BBox:", bbox)
    #         st.write("Straight QR code:", straight_qrcode)