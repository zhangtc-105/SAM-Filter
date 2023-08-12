import cv2

image_path = "color.png"
image = cv2.imread(image_path)
cv2.imshow("image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()