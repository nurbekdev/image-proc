import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# YOLO modelini yuklash
@st.cache_resource
def load_model():
    return YOLO("yolov5s.pt")  # YOLOv5 modeli

# Funksiyalar
def translate_image(image, tx, ty):
    rows, cols = image.shape[:2]
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, matrix, (cols, rows))

def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    center = (cols // 2, rows // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(image, matrix, (cols, rows))

def scale_image(image, scale_factor):
    return cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

def crop_image(image, x_start, y_start, x_end, y_end):
    return image[y_start:y_end, x_start:x_end]

def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)

def adjust_brightness_contrast(image, brightness, contrast):
    return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

def fft_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    return np.uint8(magnitude_spectrum / np.max(magnitude_spectrum) * 255)

def dct_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray) / 255.0)
    return (dct * 255).astype(np.uint8)

def detect_objects(image, model, confidence):
    results = model(image)
    detected_image = results[0].plot()
    objects = results[0].boxes.data.cpu().numpy()
    return detected_image, objects

def explain_algorithm(description):
    st.info(description)

# Streamlit interfeysi
st.title("Tasvirlarga ishlov berish algoritmlari (22 ta)")

# Tasvir yuklash
uploaded_image = st.file_uploader("Tasvirni yuklang", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = Image.open(uploaded_image)
    image_np = np.array(image)
    st.image(image_np, caption="Yuklangan tasvir", use_container_width=True)

    # Amalni tanlash
    operation = st.selectbox(
        "Amalni tanlang",
        [
            "1. Ko‘chirish (Translatsiya)",
            "2. Aylantirish (Rotatsiya)",
            "3. Masshtablash (Shkala o‘zgartirish)",
            "4. Tasvirni kesish (Cropping)",
            "5. Qirqish (Clipping) [Izoh]",
            "6. Interpolatsiya [Izoh]",
            "7. Tasvirni yaxshilash (Image Enhancement)",
            "8. Tasvirni siqish (Image Compression) [Izoh]",
            "9. Tasvirni segmentatsiyalash (Image Segmentation) [Izoh]",
            "10. Tasvirni aniqlash (Image Recognition)",
            "11. Tasvirlarni tiklash (Image Restoration) [Izoh]",
            "12. Tasvirlarni qayta o'lchash (Image Resizing) [Izoh]",
            "13. Tasvirlarni moslashtirish (Image Registration) [Izoh]",
            "14. Tasvirlarni filtrlash (Image Filtering) [Izoh]",
            "15. Tasvirlarni o‘zgartirish (Geometric Transformations) [Izoh]",
            "16. Tez Furye almashtirishlar (FFT)",
            "17. Diskret kosinus almashtirish (DCT)",
            "18. Hadamard almashtirishi [Izoh]",
            "19. Rasm yorqinligi va kontrastini lokal nochiziqli tuzatish algoritmlari [Izoh]",
            "20. Gistogrammani tekislash (Histogram Equalization)",
            "21. Chiziqli bo'lmagan kontrast [Izoh]",
            "22. Yorqinlik va kontrast xususiyatlarini lokal nochiziqli tuzatish [Izoh]"
        ]
    )

    # Parametrlar va ishlov berish
    result = None
    if operation == "1. Ko‘chirish (Translatsiya)":
        tx = st.slider("X o‘qi bo‘yicha ko‘chirish", -100, 100, 0)
        ty = st.slider("Y o‘qi bo‘yicha ko‘chirish", -100, 100, 0)
        result = translate_image(image_np, tx, ty)

    elif operation == "2. Aylantirish (Rotatsiya)":
        angle = st.slider("Burchak", 0, 360, 0)
        result = rotate_image(image_np, angle)

    elif operation == "3. Masshtablash (Shkala o‘zgartirish)":
        scale_factor = st.slider("Masshtab", 0.1, 3.0, 1.0, step=0.1)
        result = scale_image(image_np, scale_factor)

    elif operation == "4. Tasvirni kesish (Cropping)":
        x_start = st.slider("Kesish boshlanish X", 0, image_np.shape[1], 0)
        y_start = st.slider("Kesish boshlanish Y", 0, image_np.shape[0], 0)
        x_end = st.slider("Kesish tugash X", 0, image_np.shape[1], image_np.shape[1])
        y_end = st.slider("Kesish tugash Y", 0, image_np.shape[0], image_np.shape[0])
        result = crop_image(image_np, x_start, y_start, x_end, y_end)

    elif operation == "7. Tasvirni yaxshilash (Image Enhancement)":
        result = enhance_image(image_np)

    elif operation == "10. Tasvirni aniqlash (Image Recognition)":
        confidence = st.slider("Minimal ishonchlilik darajasi (%)", 0, 100, 50) / 100
        model = load_model()
        result, objects = detect_objects(image_np, model, confidence)
        st.write("Aniqlangan ob'ektlar:")
        for obj in objects:
            x1, y1, x2, y2, conf, cls = obj[:6]
            st.write(f"Ob'ekt: {int(cls)}, Ishonchlilik: {conf:.2f}")

    elif operation == "16. Tez Furye almashtirishlar (FFT)":
        result = fft_image(image_np)

    elif operation == "17. Diskret kosinus almashtirish (DCT)":
        result = dct_image(image_np)

    elif operation == "20. Gistogrammani tekislash (Histogram Equalization)":
        result = enhance_image(image_np)

    else:
        explain_algorithm("Ushbu amal uchun izoh mavjud. Faqat ishlaydigan funksiyalarni tanlang.")

    # Natijani ko‘rsatish
    if result is not None:
        st.image(result, caption="Natija", use_container_width=True)

