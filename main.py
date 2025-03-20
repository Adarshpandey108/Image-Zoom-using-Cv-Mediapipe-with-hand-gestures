import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  
cap.set(4, 720)   

detector = HandDetector(detectionCon=0.8)
StartDist = None
scale = 0
cx, cy = 500, 500  

img1 = cv2.imread('img_1.png')
if img1 is None:
    raise ValueError("Error: 'img_1.png' not found or could not be loaded.")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from webcam")
        continue  

    hands, img = detector.findHands(img)

    if len(hands) == 2:
        fingers1 = detector.fingersUp(hands[0])
        fingers2 = detector.fingersUp(hands[1])

        if fingers1 == [1, 1, 0, 0, 0] and fingers2 == [1, 1, 0, 0, 0]:
            lmList1 = hands[0]["lmList"]
            lmList2 = hands[1]["lmList"]

            if len(lmList1) > 8 and len(lmList2) > 8:
                p1 = lmList1[8][:2]  
                p2 = lmList2[8][:2]  

                length, info, img = detector.findDistance(p1, p2, img)

                if StartDist is None:
                    StartDist = length 

                
                scale = int((length - StartDist) // 2)
                scale = max(-200, min(scale, 200)) 
          
                cx, cy = info[4:]

    else:
        StartDist = None  

    h1, w1, _ = img1.shape
    newH, newW = h1 + scale, w1 + scale

    newH = max(50, (newH // 2) * 2)  
    newW = max(50, (newW // 2) * 2)

    img1_resized = cv2.resize(img1, (newW, newH))

    h, w, _ = img.shape 

    x1, x2 = max(0, cx - newW // 2), min(w, cx + newW // 2)
    y1, y2 = max(0, cy - newH // 2), min(h, cy + newH // 2)

    overlay = img1_resized[:y2 - y1, :x2 - x1]

    try:
        img[y1:y2, x1:x2] = overlay
    except ValueError:
        print("Error: Overlay size mismatch!")

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
