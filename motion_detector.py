import cv2
import datetime
import pandas

first_frame=None
status_list=[None,None]
times=[]
# use DataFrame to store the detected
# time and movement time of the object in the frame
df=pandas.DataFrame(columns=["Start","End"])

video=cv2.VideoCapture(0)
while True:
    check,frame=video.read()
    # status at the beginning of the recording is 0
    # as the object is not visible
    status=0

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)
    # save the image in a specific frame
    if first_frame is None:
        first_frame=gray
        continue

    # calculates the difference between the first frame and other frames
    delta_frame=cv2.absdiff(first_frame,gray)

    thresh_delta=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]

    thresh_delta=cv2.dilate(thresh_delta,None,iterations=0)
    # define the contour area
    (cnts,_)=cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour)<1000:
            continue
        # change in status when the object is being detected
        status=1
        # creates a rectangular box around the object in the frame
        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    # list of status for every frame
    status_list.append(status)

    status_list=status_list[-2:]

    # record datetime in a list when change occurs
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.datetime.now())

    cv2.imshow('frame',frame)
    cv2.imshow('Capturing',gray)
    cv2.imshow('delta',delta_frame)
    cv2.imshow('thresh',thresh_delta)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# store the time value in the DataFrame
# and write the DataFrame into a CSV file

    print(status_list)
    print(times)

    for i in range(0,len(times)):
    # stroe time values in a DataFrame
        df=df.append({"Start":times[i],"End":datetime.datetime.now()},ignore_index=True)
        i+=1
    # df.to_csv("times.csv")
    df.to_json("times.json")

video.release()

cv2.destroyAllWindows()

