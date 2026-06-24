import cv2
import numpy as np
image = cv2.imread("sample.jpg")
grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sobel_x = cv2.Sobel(grey_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(grey_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x.astype(np.float32), sobel_y.astype(np.float32))
sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
print(sobel_combined.dtype, sobel_combined.max())

cv2.namedWindow("Grayscale Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Sobel-X Output", cv2.WINDOW_NORMAL)
cv2.namedWindow("Sobel-Y Output", cv2.WINDOW_NORMAL)
cv2.namedWindow("combined", cv2.WINDOW_NORMAL)

cv2.imwrite("edges.png", cv2.convertScaleAbs(sobel_combined))
cv2.imshow("Grayscale Image", grey_image)
cv2.imshow("combined", cv2.convertScaleAbs(sobel_combined))
cv2.imshow("Sobel-X Output", cv2.convertScaleAbs(sobel_x))
cv2.imshow("Sobel-Y Output", cv2.convertScaleAbs(sobel_y))
cv2.waitKey(0)
cv2.destroyAllWindows()