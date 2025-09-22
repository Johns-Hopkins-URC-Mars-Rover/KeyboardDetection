import cv2
from ultralytics import YOLO

def correct_key_label(cls_name, x1, y1, x2, y2, frame, results):
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    if cls_name in ["y", "z"]:
        other_keys = []
        for r in results:
            for box in r.boxes:
                name = model.names[int(box.cls[0])]
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                bcx = (bx1 + bx2) // 2
                bcy = (by1 + by2) // 2
                other_keys.append((name, bcx, bcy))

        if cls_name == "y":
            t_key = [ox for (name, ox, oy) in other_keys if name == "t"]
            u_key = [ox for (name, ox, oy) in other_keys if name == "u"]
            if t_key and u_key:
                if min(t_key[0], u_key[0]) < cx < max(t_key[0], u_key[0]):
                    return "y"
                else:
                    return "z"

        if cls_name == "z":
            x_key = [ox for (name, ox, oy) in other_keys if name == "x"]
            if x_key and cx < x_key[0]:
                return "z"
            else:
                return "y"

    return cls_name

model = YOLO(r'/Users/joshuadayal/Downloads/best.pt')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(frame, conf=0.5, verbose=False, show=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            cls_name = cls_name = correct_key_label(model.names[cls], x1, y1, x2, y2, frame, results)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame,f'{cls_name}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
