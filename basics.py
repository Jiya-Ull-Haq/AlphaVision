
import cv2

img = cv2.imread ("/Users/shaik/PycharmProjects/img/known/screen-shot-2019-02-15-at-19-16-58-720x720.jpg", 1)

resized = cv2.resize(img,(int(img.shape[1]*2),int(img.shape[0]*2)))

cv2.imshow("Legend",resized)

cv2.waitKey(0)

cv2.destroyAllWindows()