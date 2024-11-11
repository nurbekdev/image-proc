import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
from scipy.signal import wiener  # Wiener filtri uchun



# YOLO modelini yuklash
@st.cache_resource
def load_model():
    return YOLO("yolov5s.pt")  # YOLOv5 modeli

# Funksiyalar
def translate_image(image, tx, ty):
    rows, cols = image.shape[:2]
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, matrix, (cols, rows))


def apply_clahe(image, clip_limit, grid_size):
    # Agar rangli tasvir bo'lsa, grayscale formatga o'tkazamiz
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # CLAHE obyekti yaratish
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    result = clahe.apply(gray)

    return result

def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    center = (cols // 2, rows // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(image, matrix, (cols, rows))

def scale_image(image, scale_factor):
    return cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

def crop_image(image, x_start, y_start, x_end, y_end):
    return image[y_start:y_end, x_start:x_end]

def clip_image(image, min_val, max_val):
    return np.clip(image, min_val, max_val)

def interpolate_image(image, scale_factor, method):
    interpolation = cv2.INTER_LINEAR if method == "Linear" else cv2.INTER_NEAREST
    return cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=interpolation)

def enhance_image(image):
    if len(image.shape) == 3:  # Rangli tasvir
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:  # Grayscale tasvir
        return cv2.equalizeHist(image)

def compress_image(image, quality):
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def hadamard_transform(image):
    st.info("Hadamard almashtirishi (Hadamard Transform) bu matematik vosita bo'lib, tasvirlarni, signallarni yoki ma'lumotlarni tez va samarali qayta ishlashda qo'llaniladi. Bu transformatsiya signallarni ortogonal asosda ifodalash uchun ishlatiladi va ko'pincha tasvirlarni siqish, ma'lumotlarni filtrlash va shovqinni kamaytirish kabi sohalarda qo'llaniladi.")
    # Tasvir formatini tekshirish
    if len(image.shape) == 3:  # Rangli tasvir
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Kulrang formatga o'tkazish

    # Tasvir hajmini 2^n o'lchamga keltirish
    h, w = image.shape
    size = 2 ** int(np.floor(np.log2(min(h, w))))  # Eng katta 2^n bo'lgan o'lcham
    resized_image = cv2.resize(image, (size, size)).astype(np.float32)

    # Hadamard matritsani yaratish
    H = hadamard(size)

    # Hadamard transformasini hisoblash
    hadamard_transformed = H @ resized_image @ H.T

    # Invers transform yordamida tiklash
    inverse_hadamard = H.T @ hadamard_transformed @ H

    return resized_image, hadamard_transformed, inverse_hadamard


def segment_image(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary

def detect_objects(image, model, confidence):
    results = model(image)
    detected_image = results[0].plot()
    objects = results[0].boxes.data.cpu().numpy()
    return detected_image, objects

def restore_image(image):
    st.info("Tasvirlarni tiklash (Image Restoration) bu buzilgan yoki shovqin qo'shilgan tasvirni qayta ishlash orqali uning asl yoki ideal ko'rinishini tiklash jarayonidir. Bu jarayon tasvir sifatini yaxshilashga qaratilgan bo'lib, u matematik va statistik modellarga asoslangan.")
    # Rangli tasvirni kulrang tasvirga aylantirish
    if len(image.shape) == 3:  # Rangli tasvir
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Shovqin qo'shish
    noisy_image = image + np.random.normal(0, 25, image.shape).astype(np.float32)

    # Gaussian filtr bilan shovqinni kamaytirish
    gaussian_filtered = cv2.GaussianBlur(noisy_image, (5, 5), 0)

    # Median filtr bilan shovqinni kamaytirish
    median_filtered = cv2.medianBlur(noisy_image.astype(np.uint8), 5)

    # Wiener filtr yordamida shovqinni kamaytirish
    wiener_filtered = wiener(noisy_image, (5, 5))

    return noisy_image, gaussian_filtered, median_filtered, wiener_filtered


def resize_image(image, width, height):
    return cv2.resize(image, (width, height))

def register_images(image1, image2):
    # ORB xususiyat detektori
    orb = cv2.ORB_create()

    # Xususiyatlarni aniqlash va tavsiflash
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

    # BF matcher yordamida mos keladigan nuqtalarni topish
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Moslikni saralash
    matches = sorted(matches, key=lambda x: x.distance)

    # Eng yaxshi mosliklarni olish (masalan, 50 ta)
    good_matches = matches[:50]

    # Mos keladigan nuqtalarni ajratib olish
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Geometrik transformatsiyani aniqlash (Homography)
    matrix, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)

    # Moslashtirish uchun tasvirni o'zgartirish
    height, width = image1.shape
    aligned_image = cv2.warpPerspective(image2, matrix, (width, height))

    return aligned_image

def filter_image(image, kernel_type):
    if kernel_type == "Blur":
        return cv2.GaussianBlur(image, (5, 5), 0)
    elif kernel_type == "Sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    return image

def geometric_transformation(image, tx, ty, angle, scale):
    rows, cols = image.shape[:2]
    trans_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    transformed = cv2.warpAffine(image, trans_matrix, (cols, rows))
    rot_matrix = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, scale)
    transformed = cv2.warpAffine(transformed, rot_matrix, (cols, rows))
    return transformed

def fft_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    return np.uint8(magnitude_spectrum / np.max(magnitude_spectrum) * 255)

def dct_image(image):
    st.info("Diskret Kosinus Almashtirish (DCT) raqamli signal va tasvirlarni siqish, filtrlash va o'zgarishlar uchun keng qo'llaniladigan matematik transformatsiyadir. DCT tasvir va signallarni chastotaviy domenlarda ifodalashda ishlatiladi. U diskret Fourier transform (DFT) ga yaqin, ammo faqat kosinus funktsiyalaridan foydalanadi, bu esa uni ko'proq siqish algoritmlarida va shovqinni kamaytirishda samarali qiladi.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray) / 255.0)
    return (dct * 255).astype(np.uint8)

# def hadamard_transform(image):
#     st.info("Hadamard transformatsiyasi qiyinchiliklar tufayli hali kiritilmagan.")

def nonlinear_contrast(image):
    lut = np.array([255 * (i / 255) ** 0.5 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, lut)

def histogram_equalization(image):
    if len(image.shape) == 3:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        return cv2.equalizeHist(image)

# Streamlit interfeysi
st.title("Tasvirlarga ishlov berish algoritmlari (22 ta)")

uploaded_image = st.file_uploader("Tasvirni yuklang", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = Image.open(uploaded_image)
    image_np = np.array(image)
    st.image(image_np, caption="Yuklangan tasvir", use_container_width=True)

    operation = st.selectbox(
        "Amalni tanlang",
        [
            "1. Ko‘chirish (Translatsiya)",
            "2. Aylantirish (Rotatsiya)",
            "3. Masshtablash (Shkala o‘zgartirish)",
            "4. Tasvirni kesish (Cropping)",
            "5. Qirqish (Clipping)",
            "6. Interpolatsiya",
            "7. Tasvirni yaxshilash (Image Enhancement)",
            "8. Tasvirni siqish (Image Compression)",
            "9. Tasvirni segmentatsiyalash (Image Segmentation)",
            "10. Tasvirni aniqlash (Image Recognition)",
            "11. Tasvirlarni tiklash (Image Restoration)",
            "12. Tasvirlarni qayta o'lchash (Image Resizing)",
            "13. Tasvirlarni moslashtirish (Image Registration)",
            "14. Tasvirlarni filtrlash (Image Filtering)",
            "15. Tasvirlarni o‘zgartirish (Geometric Transformations)",
            "16. Tez Furye almashtirishlar (FFT)",
            "17. Diskret kosinus almashtirish (DCT)",
            "18. Hadamard transformatsiyasi",
            "19. Chiziqli bo‘lmagan kontrast",
            "20. Gistogrammani tekislash",
            "21. Yorqinlik va kontrast xususiyatlarini lokal nochiziqli tuzatish algoritmlari."
        ]
    )

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


    elif operation == "21. Yorqinlik va kontrast xususiyatlarini lokal nochiziqli tuzatish algoritmlari.":
        clip_limit = st.slider("Clip Limit", 1.0, 10.0, 2.0, step=0.1)
        grid_size = st.slider("Tile Grid Size", 1, 16, 8)
        result = apply_clahe(image_np, clip_limit, grid_size)


    elif operation == "5. Qirqish (Clipping)":
        min_val = st.slider("Minimal qiymat", 0, 255, 0)
        max_val = st.slider("Maksimal qiymat", 0, 255, 255)
        result = clip_image(image_np, min_val, max_val)

    elif operation == "6. Interpolatsiya":
        scale_factor = st.slider("Masshtab koeffitsiyenti", 0.1, 3.0, 1.0)
        method = st.radio("Interpolatsiya usuli", ["Linear", "Nearest"])
        result = interpolate_image(image_np, scale_factor, method)

    elif operation == "7. Tasvirni yaxshilash (Image Enhancement)":
        result = enhance_image(image_np)

    elif operation == "8. Tasvirni siqish (Image Compression)":
        quality = st.slider("Siqish sifati", 10, 100, 85)
        result = compress_image(image_np, quality)

    elif operation == "9. Tasvirni segmentatsiyalash (Image Segmentation)":
        threshold = st.slider("Threshold", 0, 255, 127)
        result = segment_image(image_np, threshold)

    elif operation == "10. Tasvirni aniqlash (Image Recognition)":
        confidence = st.slider("Minimal ishonchlilik darajasi (%)", 0, 100, 50) / 100
        model = load_model()
        result, objects = detect_objects(image_np, model, confidence)
        st.write("Aniqlangan ob'ektlar:")
        for obj in objects:
            x1, y1, x2, y2, conf, cls = obj[:6]
            st.write(f"Ob'ekt: {int(cls)}, Ishonchlilik: {conf:.2f}")


    elif operation == "11. Tasvirlarni tiklash (Image Restoration)":

        # Tasvirni tiklash

        noisy_image, gaussian_filtered, median_filtered, wiener_filtered = restore_image(image_np)

        # Natijalarni ko'rsatish

        st.subheader("Tasvirni Tiklash Natijalari")

        col1, col2 = st.columns(2)

        col3, col4 = st.columns(2)

        with col1:

            st.image(noisy_image, caption="Shovqinli Tasvir", use_container_width=True, clamp=True)

        with col2:

            st.image(gaussian_filtered, caption="Gaussian Filtr", use_container_width=True, clamp=True)

        with col3:

            st.image(median_filtered, caption="Median Filtr", use_container_width=True, clamp=True)

        with col4:

            st.image(wiener_filtered, caption="Wiener Filtr", use_container_width=True, clamp=True)


    elif operation == "12. Tasvirlarni qayta o'lchash (Image Resizing)":
        width = st.number_input("Kenglik (px)", min_value=1, value=image_np.shape[1])
        height = st.number_input("Balandlik (px)", min_value=1, value=image_np.shape[0])
        result = resize_image(image_np, int(width), int(height))


    elif operation == "13. Tasvirlarni moslashtirish (Image Registration)":

        st.info("Iltimos, ikkita tasvirni yuklang.")

        uploaded_image1 = st.file_uploader("Maqsad tasvirni yuklang (Tasvir 1)", type=["jpg", "jpeg", "png"],
                                           key="image1")

        uploaded_image2 = st.file_uploader("Moslashtiriladigan tasvirni yuklang (Tasvir 2)",
                                           type=["jpg", "jpeg", "png"], key="image2")

        if uploaded_image1 and uploaded_image2:
            image1 = Image.open(uploaded_image1).convert("L")  # Maqsad tasvir (kulrang formatda)

            image2 = Image.open(uploaded_image2).convert("L")  # Moslashtiriladigan tasvir (kulrang formatda)

            image1_np = np.array(image1)

            image2_np = np.array(image2)

            # Moslashtirish

            aligned_image = register_images(image1_np, image2_np)

            # Natijalarni ko'rsatish

            st.subheader("Tasvirlarni Moslashtirish Natijalari")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.image(image1_np, caption="Maqsad Tasvir", use_container_width=True, clamp=True)

            with col2:
                st.image(image2_np, caption="Moslashtiriladigan Tasvir", use_container_width=True, clamp=True)

            with col3:
                st.image(aligned_image, caption="Moslashtirilgan Tasvir", use_container_width=True, clamp=True)


    elif operation == "14. Tasvirlarni filtrlash (Image Filtering)":
        kernel_type = st.radio("Filtr turi", ["Blur", "Sharpen"])
        result = filter_image(image_np, kernel_type)

    elif operation == "15. Tasvirlarni o‘zgartirish (Geometric Transformations)":
        tx = st.slider("X o‘qi bo‘yicha ko‘chirish", -100, 100, 0)
        ty = st.slider("Y o‘qi bo‘yicha ko‘chirish", -100, 100, 0)
        angle = st.slider("Burchak", 0, 360, 0)
        scale = st.slider("Masshtab", 0.1, 3.0, 1.0)
        result = geometric_transformation(image_np, tx, ty, angle, scale)

    elif operation == "16. Tez Furye almashtirishlar (FFT)":
        result = fft_image(image_np)

    elif operation == "17. Diskret kosinus almashtirish (DCT)":
        result = dct_image(image_np)


    elif operation == "18. Hadamard transformatsiyasi":

        # Hadamard transformasini bajarish

        original, transformed, restored = hadamard_transform(image_np)

        # Natijalarni ko'rsatish

        st.subheader("Hadamard Transformatsiyasi Natijalari")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.image(original, caption="Original Tasvir", use_container_width=True, clamp=True)

        with col2:

            st.image(np.log1p(np.abs(transformed)), caption="Hadamard Transformasi", use_container_width=True,
                     clamp=True)

        with col3:

            st.image(restored, caption="Tiklangan Tasvir", use_container_width=True, clamp=True)

    elif operation == "19. Chiziqli bo‘lmagan kontrast":
        result = nonlinear_contrast(image_np)

    elif operation == "20. Gistogrammani tekislash":
        result = histogram_equalization(image_np)

    # elif operation == "21. Yorqinlik va kontrast xususiyatlarini lokal nochiziqli tuzatish":
    #     st.info("Yorqinlik va kontrastni lokal tarzda tuzatish izoh berilgan.")

    if result is not None:
        st.image(result, caption="Natija", use_container_width=True)
