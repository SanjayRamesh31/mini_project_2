import cv2 
image = cv2.imread('image.png') 
h, w = image.shape[:2] 
print("Height = {}, Width = {}".format(h, w)) 





# (B, G, R) = image[100, 100] 
# print("R = {}, G = {}, B = {}".format(R, G, B)) 
# B = image[100, 100, 0] 
# print("B = {}".format(B)) 



# roi = image[100 : 500, 200 : 700] 


# resize = cv2.resize(image, (800, 800)) 




# ratio = 800 / w 
# dim = (800, int(h * ratio)) 
# resize_aspect = cv2.resize(image, dim) 
