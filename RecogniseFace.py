import numpy as np
from PrepareDataset import *
import AddNewFace
import cv2
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
import pickle


images = []
labels = []
labels_dic = {}

choice = input("Do you want to add new face? (Yes or No) ")
if choice == 'yes':
    AddNewFace.add_face()


def collect_dataset():

    people = [person for person in os.listdir("people/")]

    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir("people/" + person):
            if image.endswith('.jpg'):
                images.append(cv2.imread("people/" + person + '/' + image, 0))
                labels.append(i)
    return images, np.array(labels), labels_dic


images, labels, labels_dic = collect_dataset()

X_train = np.asarray(images)
train = X_train.reshape(len(X_train), -1)

sc = StandardScaler()
X_train_sc = sc.fit_transform(train.astype(np.float64))
pca1 = PCA(n_components=.97)
new_train = pca1.fit_transform(X_train_sc)
kf = KFold(n_splits=5,shuffle=True)
param_grid = {'C': [.0001, .001, .01, .1, 1, 10]}
gs_svc = GridSearchCV(SVC(kernel='linear', probability=True), param_grid=param_grid, cv=kf, scoring='accuracy')
gs_svc.fit(new_train, labels)
clf = gs_svc.best_estimator_
filename = 'svc_linear_face.pkl'
f = open(filename, 'wb')
pickle.dump(clf, f)
f.close()

filename = 'svc_linear_face.pkl'
svc1 = pickle.load(open(filename, 'rb'))

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
cv2.namedWindow("opencv_face ", cv2.WINDOW_AUTOSIZE)

while True:
    ret, frame = cam.read()

    faces_coord = detect_face(frame)  # detect more than one face
    if len(faces_coord):
        faces = normalize_faces(frame, faces_coord)

        for i, face in enumerate(faces):  # for each detected face

            t = face.reshape(1, -1)
            t = sc.transform(t.astype(np.float64))
            test = pca1.transform(t)
            prob = svc1.predict_proba(test)
            confidence = svc1.decision_function(test)
            print(confidence)
            print(prob)

            pred = svc1.predict(test)
            print(pred, pred[0])

            name = labels_dic[pred[0]].capitalize()
            print(name)

            cv2.putText(frame, name, (faces_coord[i][0], faces_coord[i][1] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 2, (66, 53, 243), 2)

        draw_rectangle(frame, faces_coord)  # rectangle around face

    cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2,
                cv2.LINE_AA)

    cv2.imshow("opencv_face", frame)  # live feed in external
    if cv2.waitKey(5) == 27:
        break

cam.release()
cv2.destroyAllWindows()
