import cv2
import numpy as np

class TennisBalle:
    def __init__(self):
        self.cap = cv2.VideoCapture(1)
        self.capp = cv2.VideoCapture(0)

    def detect(self):
        while True:
            ret, frame = self.cap.read()

            brightness_matrix = np.ones(frame.shape, dtype="uint8") * 60 #adjust this value for darkening and find the best option
                                                                         #hopefully raspberry pis do not have auto brightening camera features
            frame = cv2.subtract(frame, brightness_matrix)

            blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)

            filtered_frame = cv2.bilateralFilter(blurred_frame, 9, 75, 75)

            hsv = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2HSV)

            lower_green = np.array([25, 50, 105]) #changing the values to suit the tennis ball would be necessary
            upper_green = np.array([70, 255, 255]) #same

            mask = cv2.inRange(hsv, lower_green, upper_green)
            mask = cv2.dilate(mask, None, iterations=2)
            mask = cv2.erode(mask, None, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=30, minRadius=10, maxRadius=100)

            if(circles is not None):
                continue
            else:
                detected = []

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 100:
                        perimeter = cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                        
                        if len(approx) > 6: #change this value for detected verticies
                            k = cv2.isContourConvex(approx)
                            if k:
                                detected.append(contour)
                                (x, y), radius = cv2.minEnclosingCircle(contour)
                                center = (int(x), int(y))
                                radius = int(radius)
                                cv2.circle(frame, center, radius, (0, 255, 0), 2)
                            else:
                                detected.append(contour)
                                (x, y), radius = cv2.minEnclosingCircle(contour)
                                center = (int(x), int(y))
                                radius = int(radius)
                                cv2.circle(frame, center, radius, (0, 255, 0), 2)
            if(len(detected) > 1):
                continue #distance detection code
            elif(len(detected) == 1):
                d = detected[0]
                (x, y), radius = cv2.minEnclosingCircle(d)
                

            cv2.imshow('tennisdetect', frame)

            ret, frame = self.capp.read()

            brightness_matrix = np.ones(frame.shape, dtype="uint8") * 60
            frame = cv2.subtract(frame, brightness_matrix)

            blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)

            filtered_frame = cv2.bilateralFilter(blurred_frame, 9, 75, 75)

            hsv = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2HSV)

            lower_color = np.array([0, 0, 0]) #color
            upper_color = np.array([0, 0, 0]) #color

            mask = cv2.inRange(hsv, lower_color, upper_color)
            mask = cv2.dilate(mask, None, iterations=2)
            mask = cv2.erode(mask, None, iterations=1)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                    
                    if len(approx) == 4:
                        #color = np.array([0, 0, 0])
                        #cv2.rectangle(frame, x - radius, y + radius, color)
                        print("basket")


            cv2.imshow('basketdetect', frame)

            if cv2.waitKey(1) & 0xFF ==ord('q') :
                break



        cv2.destroyAllWindows()

if __name__ == "__main__":
    tb = TennisBalle()
    tb.detect()
  
