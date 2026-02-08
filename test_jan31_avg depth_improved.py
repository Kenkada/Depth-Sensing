import cv2
import numpy as np
import mediapipe as mp

# ================= LOAD CALIBRATION =================
data = np.load("stereo_params_main.npz")

K1, D1 = data["K1"], data["D1"]
K2, D2 = data["K2"], data["D2"]
R1, R2 = data["R1"], data["R2"]
P1, P2 = data["P1"], data["P2"]
T = data["T"]

# ================= MIN DISTANCE THRESHOLD =================
baseline = np.linalg.norm(T)          # meters
MIN_DISTANCE = 5 * baseline           # meters
MIN_DISTANCE_CM = MIN_DISTANCE * 100  # cm

# ================= OPEN CAMERAS =================
capL = cv2.VideoCapture(2, cv2.CAP_DSHOW)
capR = cv2.VideoCapture(1, cv2.CAP_DSHOW)

capL.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

for _ in range(20):
    capL.read()
    capR.read()

ret, frameL = capL.read()
h, w = frameL.shape[:2]

# ================= RECTIFICATION MAPS =================
map1x, map1y = cv2.initUndistortRectifyMap(
    K1, D1, R1, P1[:3, :3], (w, h), cv2.CV_32FC1
)
map2x, map2y = cv2.initUndistortRectifyMap(
    K2, D2, R2, P2[:3, :3], (w, h), cv2.CV_32FC1
)

# ================= MEDIAPIPE =================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0)

# ================= WINDOWS =================
cv2.namedWindow("Left Input")
cv2.namedWindow("Left Overlay")
cv2.namedWindow("Right Input")
cv2.namedWindow("Right Overlay")

screen_x, screen_y = 300, 200
cv2.moveWindow("Left Input", screen_x, screen_y)
cv2.moveWindow("Right Input", screen_x + w + 20, screen_y)
cv2.moveWindow("Left Overlay", screen_x, screen_y + h + 40)
cv2.moveWindow("Right Overlay", screen_x + w + 20, screen_y + h + 40)

print("Stereo depth sensing running. Press 'q' to quit.")

# ================= MAIN LOOP =================
while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or not retR:
        break

    rectL = cv2.remap(frameL, map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(frameR, map2x, map2y, cv2.INTER_LINEAR)

    overlayL = np.zeros_like(rectL)
    overlayR = np.zeros_like(rectR)

    resL = pose.process(cv2.cvtColor(rectL, cv2.COLOR_BGR2RGB))
    resR = pose.process(cv2.cvtColor(rectR, cv2.COLOR_BGR2RGB))

    depths = []

    if resL.pose_landmarks and resR.pose_landmarks:
        for i, lmL in enumerate(resL.pose_landmarks.landmark):
            lmR = resR.pose_landmarks.landmark[i]

            if lmL.visibility > 0.6 and lmR.visibility > 0.6:
                xL, yL = int(lmL.x * w), int(lmL.y * h)
                xR, yR = int(lmR.x * w), int(lmR.y * h)

                ptsL = np.array([[xL], [yL]], dtype=np.float32)
                ptsR = np.array([[xR], [yR]], dtype=np.float32)

                point_4d = cv2.triangulatePoints(P1, P2, ptsL, ptsR)
                point_3d = point_4d[:3] / point_4d[3]

                Z_cm = point_3d[2, 0] * 100  # cm
                depths.append(Z_cm)

                cv2.circle(overlayL, (xL, yL), 5, (0, 0, 255), -1)
                cv2.circle(overlayR, (xR, yR), 5, (0, 0, 255), -1)

                cv2.putText(overlayL, f"{Z_cm:.1f}",
                            (xL + 5, yL - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

                cv2.putText(overlayR, f"{Z_cm:.1f}",
                            (xR + 5, yR - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    # ================= UPPER DISPLAY =================
    if depths:
        avg_depth_cm = np.mean(depths)

        status = "OK" if avg_depth_cm > MIN_DISTANCE_CM else "TOO CLOSE"

        cv2.putText(rectL,
                    f"Min Dist: {MIN_DISTANCE_CM:.1f} cm",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(rectL,
                    f"Person: {avg_depth_cm:.1f} cm [{status}]",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if status == "OK" else (0, 0, 255), 2)

        cv2.putText(rectR,
                    f"Person: {avg_depth_cm:.1f} cm",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Left Input", rectL)
    cv2.imshow("Left Overlay", overlayL)
    cv2.imshow("Right Input", rectR)
    cv2.imshow("Right Overlay", overlayR)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capL.release()
capR.release()
cv2.destroyAllWindows()
