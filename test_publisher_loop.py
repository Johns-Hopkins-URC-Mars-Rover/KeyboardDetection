import cv2
import time

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open camera")
        return

    # Match your ROS publisher settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    prev = time.time()
    frame_count = 0
    fps_accumulator = 0.0

    while True:
        ret, frame = cap.read()
        now = time.time()

        if not ret:
            print(" Frame grab failed")
            break

        # Calculate instantaneous FPS
        fps = 1.0 / (now - prev)
        prev = now

        frame_count += 1
        fps_accumulator += fps

        # Print stats every 30 frames (~1 second)
        if frame_count % 30 == 0:
            avg_fps = fps_accumulator / 30
            data_size = frame.nbytes
            print(f"[Frame {frame_count}] "
                  f"AvgFPS={avg_fps:.2f}, "
                  f"Shape={frame.shape}, "
                  f"Bytes={data_size}")
            fps_accumulator = 0.0  # reset

        cv2.imshow("Simulated ROS Publisher Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()