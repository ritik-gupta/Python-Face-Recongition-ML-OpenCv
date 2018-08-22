from PrepareDataset import *
import os


def add_face():
    cam = cv2.VideoCapture(0)

    folder = "people/" + input('Person:').lower()
    try:
        os.mkdir(folder)

        flag_start_capturing = False
        sample = 1
        cv2.namedWindow("Face", cv2.WINDOW_AUTOSIZE)

        while True:
            ret, frame = cam.read()

            faces_coord = detect_face(frame)

            if len(faces_coord):
                faces = normalize_faces(frame, faces_coord)
                cv2.imwrite(folder + '/' + str(sample) + '.jpg', faces[0])

                if flag_start_capturing:
                    sample += 1

            draw_rectangle(frame, faces_coord)
            cv2.imshow('Face', frame)
            keypress = cv2.waitKey(1)

            if keypress == ord('c'):

                if not flag_start_capturing:
                    flag_start_capturing = True

            if sample > 150:
                break

        cam.release()
        cv2.destroyAllWindows()
    except FileExistsError:
        print("Already exists")

