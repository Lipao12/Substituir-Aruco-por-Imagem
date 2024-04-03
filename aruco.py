import cv2
import numpy as np
from cv2 import aruco

image_to_replace = cv2.imread("aruco\image_to_replace\Leonardo-Mona-Lisa.jpg")

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters()

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()

    if not ret or frame is None:
        print("End of Video")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, _ = aruco.detectMarkers(gray_frame, aruco_dict, parameters=parameters)
    #print(marker_corners)
    if marker_IDs is not None:
        for i in range(len(marker_IDs)):
            corners = marker_corners[i].squeeze().astype(np.int32)

            if image_to_replace is not None:
                pts_src = np.array([[0, 0], [0, image_to_replace.shape[0]], [image_to_replace.shape[1], image_to_replace.shape[0]],
                     [image_to_replace.shape[1], 0]], dtype=np.float32)
                pts_dst = corners.astype(np.float32)
                M, _ = cv2.findHomography(pts_src, pts_dst)
                h, w = image_to_replace.shape[:2]
                warped_image = cv2.warpPerspective(image_to_replace, M, (frame.shape[1], frame.shape[0]))
                mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255
                cv2.fillPoly(mask, [corners], 0)
                masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
                frame = cv2.add(masked_frame, warped_image)

    cv2.imshow('output', frame)
    if cv2.waitKey(1) == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
