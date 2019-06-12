import cv2
 
class ShapeDetector:
    def __init__(self):
        pass
 
#	def detect(self, c):
#		# initialize the shape name and approximate the contour
#		shape = "unidentified"
#        #arcLength computes the perimeter of the shape c. True is for a closed shape
#		peri = cv2.arcLength(c, True)
#        #Approximates the shape c with less vertices, with accuracy parameter as the second argument
#        #Here, the accuracy is a variation of 4% of the perimeter
#        #the resulting shape is the output of the function
#		approx = cv2.approxPolyDP(c, 0.04 * peri, True)
#		if len(approx) == 3:
#			shape = "triangle"
#		elif len(approx) == 4:
#			(x, y, w, h) = cv2.boundingRect(approx)
#			ar = w / float(h)
#			shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
#		elif len(approx) == 5:
#			shape = "pentagon"
#		else:
#			shape = "circle"
#		return shape

    def detect(self,c):
        shape="unidentified"
        peri = cv2.arcLength(c,True)
        area = cv2.contourArea(c)
        (x,y),radius=cv2.minEnclosingCircle(c)
#        center=(int(x),int(y))
        radius=int(radius)
        if area!=0 and peri !=0:
#            print(peri**2/area-4*3.14159)
#            if abs(abs(peri**2/area)-4*3.14159)/(peri**2/area)<=0.01:
            if abs(abs(area)-3.14159*radius**2)/area<=0.10:
#            if abs(abs(peri**2/area)-4*3.14159)/(4*3.14159)<=1.0:
                shape="circle"
        return shape