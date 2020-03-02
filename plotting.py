
import cv2

video = cv2.VideoCapture(0)

a=1

while True:
    a = a + 1
    Check, frame = video.read()
    print (frame)

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    cv2.imshow("capturing",gray)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
print(a)
video.release()
cv2.destroyAllWindows()