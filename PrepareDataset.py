import cv2


def detect_face(frame):

        if frame.ndim != 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detector = cv2.CascadeClassifier("xml/frontal_face.xml")

        faces = detector.detectMultiScale(frame, 1.2, 5)

        return faces


def gray_scale(image):
        if image.ndim != 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image


def cut_faces(image, faces_coord):
        faces = []

        for (x, y, w, h) in faces_coord:
            faces.append(image[y: y + h, x: x + w])

        return faces


def normalize_intensity(images):
        images_norm = []
        for image in images:
            is_color = len(image.shape) == 3
            if is_color:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images_norm.append(cv2.equalizeHist(image))
        return images_norm


def resize(images, size=(47, 62)):
        image_resize = []

        for image in images:
            if image.shape < size:
                img_size = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
            else:
                img_size = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            image_resize.append(img_size)

        return image_resize


def normalize_faces(frame, faces_coord):
    gray_frame = gray_scale(frame)
    faces = cut_faces(gray_frame, faces_coord)
    faces = normalize_intensity(faces)

    faces = resize(faces)
    return faces


def draw_rectangle(image, coords):
    for (x, y, w, h) in coords:

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
