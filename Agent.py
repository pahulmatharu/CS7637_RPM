







# pylint: disable=C0200,C0116,C0303,C0325,C0103,C0115

# Allowable libraries:
# - Python 3.10.12
# - Pillow 10.0.0
# - numpy 1.25.2
# - OpenCV 4.6.0 (with opencv-contrib-python-headless 4.6.0.66)

# To activate image processing, uncomment the following imports:
from PIL import Image
import numpy as np
import cv2

A = 'A'
B = 'B'
C = 'C'
D = 'D'
E = 'E'
F = 'F'
G = 'G'
H = 'H'
figure_index_1 = '1'
figure_index_2 = '2'
figure_index_3 = '3'
figure_index_4 = '4'
figure_index_5 = '5'
figure_index_6 = '6'
figure_index_7 = '7'
figure_index_8 = '8'
ROTATE = 'rotate'
FLIP = 'flip'
SAME = 'same'
FILL_WHOLE_IMG = 'FILL_WHOLE_IMG'

class Agent:
    def __init__(self):
        self.TwoByTwo = '2x2'
        """
        The default constructor for your Agent. Make sure to execute any processing necessary before your Agent starts
        solving problems here. Do not add any variables to this signature; they will not be used by main().
        
        This init method is only called once when the Agent is instantiated 
        while the Solve method will be called multiple times. 
        """
        pass
    
    def similarity(self, image, template):
        gImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gTemplate = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        return (cv2.matchTemplate(gImage, gTemplate, cv2.TM_CCOEFF_NORMED).max()) * 100  
    
    def denoise_image(self, image_filename):
        image = cv2.imread(image_filename)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
        bg=cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
        out_gray=cv2.divide(image, bg, scale=255)
        out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1] 
        return out_binary
    
    def connected_components(self, image):
        # because connected components expect white on black, we need to reverse the image
        inverse = cv2.bitwise_not(image)
        
        ret, thresh = cv2.threshold(inverse,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # You need to choose 4 or 8 for connectivity type
        connectivity = 4  
        # Perform the operation
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
        
        # Shape count: num_labels variable will contain the total number of shapes in the image, including the background. So subtract 1 
        shape_count = num_labels - 1
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            print("Shape {}:".format(i))
            print("  Left: {}".format(x))
            print("  Top: {}".format(y))
            print("  Width: {}".format(w))
            print("  Height: {}".format(h))
            print("  Area: {}".format(area))
        
        label_hue = np.uint8(179*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # cvt to BGR for display
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # set bg label to black
        labeled_img[label_hue==0] = 0

        cv2.imshow('labeled.png', labeled_img)
        cv2.waitKey()
            # cv2.destroyAllWindows()
        print(num_labels, stats, centroids)
    
    def Solve(self, problem):
        # Primary method for solving incoming Raven's Progressive Matrices.

        # Args:
        #     problem: The RavensProblem instance.

        # Returns:
        #     int: The answer (1-6 for 2x2 OR 1-8 for 3x3).
        #     Return a negative number to skip a problem.
        #     Remember to return the answer [Key], not the name, as the ANSWERS ARE SHUFFLED in Gradescope.
        # DO NOT use absolute file pathing to open files.

        # Example: Read the 'A' figure from the problem using Pillow
        #     image_a = Image.open(problem.figures["A"].visualFilename)

        # Example: Read the '1' figure from the problem using OpenCv
        #     image_1 = cv2.imread(problem.figures["1"].visualFilename)

        # Don't forget to uncomment the import!
        name = problem.name
        problemType = problem.problemType
        problemSetName = problem.problemSetName
        hasVisual = problem.hasVisual
        hasVerbal = problem.hasVerbal
        figures = problem.figures

        print('Beginning problem' + name, problemType)
        # First I need to figure out what type of matrix i will need to build
        is2x2 = problemType == self.TwoByTwo
        if(not is2x2):
            # 3x3
            # Question
            ImageA_filename = figures[A].visualFilename
            ImageB_filename = figures[B].visualFilename
            ImageC_filename = figures[C].visualFilename
            ImageD_filename = figures[D].visualFilename
            ImageE_filename = figures[E].visualFilename
            ImageF_filename = figures[F].visualFilename
            ImageG_filename = figures[G].visualFilename
            ImageH_filename = figures[H].visualFilename

            #Answers
            answerImg_filename1 = figures[figure_index_1].visualFilename
            answerImg_filename2 = figures[figure_index_2].visualFilename
            answerImg_filename3 = figures[figure_index_3].visualFilename
            answerImg_filename4 = figures[figure_index_4].visualFilename
            answerImg_filename5 = figures[figure_index_5].visualFilename
            answerImg_filename6 = figures[figure_index_6].visualFilename
            answerImg_filename7 = figures[figure_index_7].visualFilename
            answerImg_filename8 = figures[figure_index_8].visualFilename
            
            image = self.denoise_image(ImageA_filename)
            self.connected_components(image)
            imageD = self.denoise_image(ImageD_filename)
            self.connected_components(imageD)
            imageG = self.denoise_image(ImageG_filename)
            self.connected_components(imageG)
            # methods


        return 1
    
    
class GeneratedAnswer:
    def __init__(self, similarity, method, answerIndex):
        self.similarity = similarity
        self.method = method
        self.answerIndex = answerIndex
        

class ShapeDetector:
    def __init__(self, image_a_filename, image_b_filename, image_c_filename):
        pass
        

    def get_shapes_count(self, filename):
        # reading image 
        img = cv2.imread(filename) 
        
        # # convert to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        
        # # setting threshold of gray image 
        _, threshold_image = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY) 
        
        # # using a findContours() function 
        contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # # print('hier', hierarchy)
        
        i = 0
        
        triangle_count = 0
        quad_count = 0
        penta_count = 0
        hexa_count = 0
        seven_count = 0
        eight_count = 0
        circle_count = 0
        # # list for storing names of shapes 
        for contour in contours: 
        #     # here we are ignoring first counter because  
        #     # findcontour function detects whole image as shape 
            if i == 0: 
                i = 1
                continue
            
            epsilon = 0.01 * cv2.arcLength(contour, True)
            # cv2.approxPloyDP() function to approximate the shape 
            approx = cv2.approxPolyDP(contour, epsilon, True) 
            
           # using drawContours() function 
            # img_cp = img.copy()
            # cv2.drawContours(img_cp, [contour], 0, (0, 0, 0), -1)
            # sim = self.sim(img_cp, img_template)
            # cv2.imshow('cp', img_cp) 
            # cv2.imshow('shapes', img) 
            # cv2.waitKey(0) 
            # cv2.destroyAllWindows()
            
            # finding center point of shape 
            x,y,w,h = cv2.boundingRect(approx)
            center_x = int(x + w/2)
            center_y = int(y + h/2)


            # putting shape name at center of each shape 
            if len(approx) == 3: 
                # triangle = Shape(center_x, center_y, 'triange')
                # shapes.append(triangle)
                triangle_count += 1
        
            elif len(approx) == 4: 
                # quad = Shape(center_x, center_y, 'quad')
                # shapes.append(quad)
                quad_count += 1
                
            elif len(approx) == 5: 
                # penta = Shape(center_x,center_y,'penta')
                # shapes.append(penta)
                penta_count += 1
                
            elif len(approx) == 6: 
                # hexa = Shape(center_x,center_y,'hexa')
                # shapes.append(hexa)
                hexa_count += 1
                
            elif len(approx) == 7: 
                # hexa = Shape(center_x,center_y,'seven')
                # shapes.append(hexa)
                seven_count += 1
                
            elif len(approx) == 8: 
                # hexa = Shape(center_x,center_y,'eight')
                # shapes.append(hexa)
                eight_count += 1
                
            else: 
                # circle = Shape(center_x,center_y, 'circle')
                # shapes.append(circle)
                circle_count += 1
                
        # for shape in shapes:
        #     print(shape.shape_type, shape.x, shape.y)
        return ShapeCount(triangle_count, quad_count, penta_count, hexa_count, seven_count, eight_count, circle_count)
            
    def sim(self, image, template):
        gImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gTemplate = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        return (cv2.matchTemplate(gImage, gTemplate, cv2.TM_CCOEFF_NORMED).max()) * 100
        

class ShapeCount:
    def __init__(self, triangle_count, quad_count,penta_count,hexa_count,seven_count,eight_count, circle_count):
        self.triangle_count = triangle_count
        self.quad_count = quad_count
        self.penta_count = penta_count
        self.hexa_count = hexa_count
        self.seven_count = seven_count
        self.eight_count = eight_count
        self.circle_count = circle_count
        self.full_count = triangle_count + quad_count + penta_count + hexa_count + seven_count + eight_count + circle_count
        
class Shape:
    def __init__(self, x, y, shape_type):
        self.x = x
        self.y = y
        self.shape_type = shape_type
        