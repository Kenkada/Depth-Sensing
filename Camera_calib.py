import cv2
import numpy as np
import sys

# ================= USER SETTINGS =================
LEFT_CAM_ID = 2
RIGHT_CAM_ID = 1

CHECKERBOARD = (9, 6)     # inner corners (columns, rows)
SQUARE_SIZE = 0.027      # meters (27 mm)

MIN_PAIRS = 20

# ================= OPEN CAMERAS =================
capL = cv2.VideoCapture(LEFT_CAM_ID, cv2.CAP_DSHOW)
capR = cv2.VideoCapture(RIGHT_CAM_ID, cv2.CAP_DSHOW)

if not capL.isOpened() or not capR.isOpened():
    print("ERROR: Cameras not opened")
    sys.exit()

capL.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# ================= WARM-UP =================
for _ in range(30):
    capL.read()
    capR.read()

# ================= WINDOW POSITIONING =================
SCREEN_W = 1920   # change if your screen resolution is different
SCREEN_H = 1080

WIN_W = 640
WIN_H = 480

left_x = (SCREEN_W // 2) - WIN_W
right_x = SCREEN_W // 2
y = (SCREEN_H - WIN_H) // 2

cv2.namedWindow("Left", cv2.WINDOW_NORMAL)
cv2.namedWindow("Right", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Left", WIN_W, WIN_H)
cv2.resizeWindow("Right", WIN_W, WIN_H)

cv2.moveWindow("Left", left_x, y)
cv2.moveWindow("Right", right_x, y)

# ================= OBJECT POINTS =================
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0],
                       0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpointsL = []
imgpointsR = []

criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    30, 0.001
)

flags = (
    cv2.CALIB_CB_ADAPTIVE_THRESH |
    cv2.CALIB_CB_FAST_CHECK |
    cv2.CALIB_CB_NORMALIZE_IMAGE
)

print("Press 's' to save a pair")
print("Press 'c' to calibrate")
print("Press 'q' to quit")

# ================= MAIN LOOP =================
while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()

    if not retL or not retR:
        print("Frame read failed")
        break

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    retCL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, flags)
    retCR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, flags)

    visL = frameL.copy()
    visR = frameR.copy()

    if retCL:
        cornersL = cv2.cornerSubPix(
            grayL, cornersL, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(visL, CHECKERBOARD, cornersL, retCL)

    if retCR:
        cornersR = cv2.cornerSubPix(
            grayR, cornersR, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(visR, CHECKERBOARD, cornersR, retCR)

    cv2.putText(
        visL,
        f"Pairs: {len(objpoints)}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Left", visL)
    cv2.imshow("Right", visR)

    key = cv2.waitKey(1) & 0xFF

    # -------- SAVE PAIR --------
    if key == ord('s'):
        if retCL and retCR:
            objpoints.append(objp)
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)
            print(f"Saved pair {len(objpoints)}")
        else:
            print("Chessboard not detected in both cameras")

    # -------- CALIBRATE --------
    if key == ord('c'):
        if len(objpoints) < MIN_PAIRS:
            print(f"Need at least {MIN_PAIRS} pairs")
            continue

        print("Calibrating...")

        _, K1, D1, _, _ = cv2.calibrateCamera(
            objpoints, imgpointsL, grayL.shape[::-1], None, None)

        _, K2, D2, _, _ = cv2.calibrateCamera(
            objpoints, imgpointsR, grayR.shape[::-1], None, None)

        _, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            objpoints,
            imgpointsL,
            imgpointsR,
            K1, D1,
            K2, D2,
            grayL.shape[::-1],
            flags=cv2.CALIB_FIX_INTRINSIC
        )

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K1, D1,
            K2, D2,
            grayL.shape[::-1],
            R, T
        )

        np.savez(
            "stereo_params.npz",
            K1=K1, D1=D1,
            K2=K2, D2=D2,
            R=R, T=T,
            R1=R1, R2=R2,
            P1=P1, P2=P2,
            Q=Q
        )

        print("Calibration complete")
        print("Baseline (cm):", np.linalg.norm(T) * 100)
        break

    if key == ord('q'):
        break

# ================= CLEANUP =================
capL.release()
capR.release()
cv2.destroyAllWindows()
