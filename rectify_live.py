import cv2
import numpy as np

# ============ Load calibration ============
data = np.load("stereo_params_main.npz")

K1, D1 = data["K1"], data["D1"]
K2, D2 = data["K2"], data["D2"]
R1, R2 = data["R1"], data["R2"]
P1, P2 = data["P1"], data["P2"]

# ============ Open cameras ============
capL = cv2.VideoCapture(2, cv2.CAP_DSHOW)
capR = cv2.VideoCapture(1, cv2.CAP_DSHOW)

capL.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# ============ Read one frame ============
retL, frameL = capL.read()
if not retL:
    print("Failed to read left camera")
    exit()

h, w = frameL.shape[:2]

# ============ Correct rectification maps ============
map1x, map1y = cv2.initUndistortRectifyMap(
    K1, D1, R1, P1[:3, :3], (w, h), cv2.CV_32FC1
)

map2x, map2y = cv2.initUndistortRectifyMap(
    K2, D2, R2, P2[:3, :3], (w, h), cv2.CV_32FC1
)

print("Rectification maps created")

# ============ Loop ============
while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        break

    rectL = cv2.remap(frameL, map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, map2x, map2y, cv2.INTER_LINEAR)

    # Draw horizontal epipolar lines
    for y in range(0, h, 30):
        cv2.line(rectL, (0, y), (w, y), (0, 255, 0), 1)
        cv2.line(rectR, (0, y), (w, y), (0, 255, 0), 1)

    cv2.imshow("Rectified Left", rectL)
    cv2.imshow("Rectified Right", rectR)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
