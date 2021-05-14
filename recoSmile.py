import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
# call camera
cap = cv2.VideoCapture(0)

while (True):
    # get the picture captured by the camera
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, 1.3, 2)
    img = frame
    for (x, y, w, h) in faces:
        # draw a face frame, blue, with a small brush width
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # select the face area in a frame, and perform eye detection in the face area 
        # instead of the entire image, saving computing resources
        face_area = img[y:y + h, x:x + w]

        ## eye detection
        # use the human eye cascade classifier engine to perform eye recognition in the face area
        # and the returned eyes is a list of eye coordinates
        eyes = eye_cascade.detectMultiScale(face_area, 1.3, 10)
        for (ex, ey, ew, eh) in eyes:
            # draw a human eye frame, green, with a brush width of 1
            cv2.rectangle(face_area, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

        ## smile detection
        # use smile cascade classifier engine to perform eye recognition in the face area
        # and the returned eyes is a list of eye coordinates

        for (ex, ey, ew, eh) in smiles:
            # draw a smile frame, red (BGR color system), with a brush width of 1
            cv2.rectangle(face_area, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)
            cv2.putText(img, 'Smile', (x, y - 7), 3, 1.2, (0, 0, 255), 2, cv2.LINE_AA)

    # real-time display effect screen
    cv2.imshow('frame2', img)
    # listen for keyboard actions every 5 milliseconds
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# finally, close all windows
cap.release()
cv2.destroyAllWindows()
