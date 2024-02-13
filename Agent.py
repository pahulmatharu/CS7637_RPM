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
    
    def IsSame2x2(self, image_one_filname, image_two_filname):
        image = cv2.imread(image_one_filname)
        image2 = cv2.imread(image_two_filname)
        imageA_array = np.array(image)
        imageB_array = np.array(image2)
        if(np.array_equal(imageA_array, imageB_array)):
            return (True, 100.0)
        
        sim = self.similarity(image, image2)
        if(sim > 99):
            return (True, sim)
        return (False, sim)       
    
    def rotate_check2x2(self, image_one_filname, image_two_filname):
        image = cv2.imread(image_one_filname)
        image2 = cv2.imread(image_two_filname)
        rotated_image90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated_image180 = cv2.rotate(image, cv2.ROTATE_180)
        rotated_image270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        valid_methods = []
        options = [(cv2.ROTATE_90_CLOCKWISE, rotated_image90), (cv2.ROTATE_180, rotated_image180), (cv2.ROTATE_90_COUNTERCLOCKWISE, rotated_image270)]
        for method in options:
            similarity = self.similarity(method[1], image2)
            # print('rotate', method[0], similarity)
            if(similarity > 70):
                valid_methods.append(('rotate', method[0], similarity))
                
        if(len(valid_methods) > 1):    
            valid_methods.sort(key=lambda tup: tup[2], reverse=True)
        return valid_methods

    def flipAxis2x2(self, image_one_filename, image_two_filename):
        image2 = cv2.imread(image_two_filename)
        image = cv2.imread(image_one_filename)
        
        horizontal_flip = cv2.flip(image, 1)
        vertical_flip = cv2.flip(image, 0)
        both_axis_flip = cv2.flip(image, -1)
        options = [(1, horizontal_flip), (0, vertical_flip), (-1, both_axis_flip)]

        valid_methods = []
        for method in options:
            similarity = self.similarity(method[1], image2)
            # print('flip', method[0], similarity)
            if(similarity > 70):
                valid_methods.append(('flip', method[0], similarity))
        
        if(len(valid_methods) > 1):    
            valid_methods.sort(key=lambda tup: tup[2], reverse=True)
        return valid_methods
    
    def generate_test_rotation2x2(self, imageC, rotationAmount, answers):
        rotated_image2 = imageC.transpose(rotationAmount)
        rotated_image2_array = np.array(rotated_image2)
        for index,answer in enumerate(answers):
            image_answer = np.array(answer)
            if(np.array_equal(rotated_image2_array, image_answer)):
                return index + 1

    
    def generate_flip2x2(self, imageC, flipDirection, answers):
        image = cv2.imread(imageC)
        flip = cv2.flip(image, flipCode=flipDirection)
        flip_arr = np.array(flip)
        for index,answer in enumerate(answers):
            answerImage = cv2.imread(answer)
            image_answer = np.array(answerImage)
            if(np.array_equal(flip_arr, image_answer)):
                return index + 1
        return None
    
    def generate_and_test(self, a_b_methods, a_c_methods, answers):
        a_b_len = len(a_b_methods)
        a_c_len = len(a_c_methods)
        if(a_b_len > 0 and a_c_len > 0):
            
        elif(a_b_len > 0):
            method = a_b_methods[0]
            id
                
        elif(a_c_len > 0):
            
        else:
            return 1
    
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
        if(is2x2):
            # images
            # figures[A].visualFilename
            # figures[B].visualFilename
            # figures[C].visualFilename
            # Answers
            answers = [figures[figure_index_1].visualFilename, 
                                figures[figure_index_2].visualFilename, 
                                figures[figure_index_3].visualFilename, 
                                figures[figure_index_4].visualFilename, 
                                figures[figure_index_5].visualFilename, 
                                figures[figure_index_6].visualFilename]
            
            # see if the relationship between A and B, A and C is a mirror.
            A_B_methods = []
            A_C_methods = []
            isSameHorinzontally = self.IsSame2x2(figures[A].visualFilename, figures[B].visualFilename)
            isSameVertically = self.IsSame2x2(figures[A].visualFilename, figures[C].visualFilename)
            if(isSameHorinzontally[0] and isSameVertically[0]):
                # most likely since both row and column are mirrors, the answer is a mirror.
                for index, answer in enumerate(answers):
                    isSame = self.IsSame2x2(figures[C].visualFilename, answer)
                    if(isSame[0]):
                        print('answer_found', index + 1)
                        return index + 1
                    
            elif(isSameHorinzontally[0]):
                A_B_methods.append(('same', 'h', isSameHorinzontally[1]))
            elif(isSameVertically[0]):
                A_C_methods.append(('same', 'v', isSameVertically[1]))
            
            rotation_methodsA_B = self.rotate_check2x2(figures[A].visualFilename, figures[B].visualFilename)
            rotation_methodsA_C = self.rotate_check2x2(figures[A].visualFilename, figures[C].visualFilename)
            # print('rotation_methods', rotation_methodsA_B, rotation_methodsA_C)
            A_B_methods.extend(rotation_methodsA_B)
            A_C_methods.extend(rotation_methodsA_C)
            # if(rotation[0]):
            #     ans = self.generate_test_rotation2x2(QuestionImageC, rotation[1], answers)
            #     if(ans is not None):
            #         print('answer_found', ans)
            #         return ans
            flip_methodsA_B = self.flipAxis2x2(figures[A].visualFilename, figures[B].visualFilename)
            flip_methodsA_C = self.flipAxis2x2(figures[A].visualFilename, figures[B].visualFilename)
            # print('flip_methods', flip_methodsA_B, flip_methodsA_C)
            A_B_methods.extend(flip_methodsA_B)
            A_C_methods.extend(flip_methodsA_C)
            # if(flip[0]):
            #     ans = self.generate_flip2x2(figures[C].visualFilename, flip[1], answer_filenames)
            #     if(ans is not None):
            #             return ans
            
            # Generate and Test
            if(len(A_B_methods) > 1):    
                A_B_methods.sort(key=lambda tup: tup[2], reverse=True)
            if(len(A_C_methods) > 1):    
                A_C_methods.sort(key=lambda tup: tup[2], reverse=True)
            answer_index = self.generate_and_test(A_B_methods, A_C_methods, answers)
            print(answer_index)
            return answer_index
        
            # Setup Image Class for each image
            # ravenImageA = RavenImage(figures[A].visualFilename)
            
        else: 
            # 3x3
            # Question
            ImageA = Image.open(figures[A].visualFilename)
            ImageB = Image.open(figures[B].visualFilename)
            ImageC = Image.open(figures[C].visualFilename)
            ImageD = Image.open(figures[D].visualFilename)
            ImageE = Image.open(figures[E].visualFilename)
            ImageF = Image.open(figures[F].visualFilename)
            ImageG = Image.open(figures[G].visualFilename)
            ImageH = Image.open(figures[H].visualFilename)

            #Answers
            Image1 = Image.open(figures[figure_index_1].visualFilename)
            Image2 = Image.open(figures[figure_index_2].visualFilename)
            Image3 = Image.open(figures[figure_index_3].visualFilename)
            Image4 = Image.open(figures[figure_index_4].visualFilename)
            Image5 = Image.open(figures[figure_index_5].visualFilename)
            Image6 = Image.open(figures[figure_index_6].visualFilename)
            Image7 = Image.open(figures[figure_index_7].visualFilename)
            Image8 = Image.open(figures[figure_index_8].visualFilename)

        # Placeholder: Skip all problems for now.
        return 1
    
    

class RavenImage:
    def __init__(self, filename):
        shapes = self.get_shapes(filename)
        self.filename = filename
        self.shapes = shapes
    
    def get_shapes(self, filename):
        # reading image 
        img = cv2.imread(filename) 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        
        # setting threshold of gray image 
        ret, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) 
        
        # using a findContours() function 
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print('hier', hierarchy)
        i = 0
        shapes = []
        # list for storing names of shapes 
        for contour in contours: 
            # here we are ignoring first counter because  
            # findcontour function detects whole image as shape 
            if i == 0: 
                i = 1
                continue
        
            # cv2.approxPloyDP() function to approximate the shape 
            approx = cv2.approxPolyDP( 
                contour, 0.01 * cv2.arcLength(contour, True), True) 
            
            # using drawContours() function 
            cv2.drawContours(img, [contour], 0, (0, 0, 255), 5) 
        
            # finding center point of shape 
            M = cv2.moments(contour) 
            if M['m00'] != 0.0: 
                x = int(M['m10']/M['m00']) 
                y = int(M['m01']/M['m00']) 
        

            # putting shape name at center of each shape 
            if len(approx) == 3: 
                triangle = Shape(x, y, 'triange')
                shapes.append(triangle)
        
            elif len(approx) == 4: 
                quad = Shape(x, y, 'quad')
                shapes.append(quad)
        
            elif len(approx) == 5: 
                penta = Shape(x,y,'penta')
                shapes.append(penta)
        
            elif len(approx) == 6: 
                hexa = Shape(x,y,'hexa')
                shapes.append(hexa)
        
            else: 
                circle = Shape(x,y, 'circle')
                shapes.append(circle)
                
        cv2.imshow('shapes', img) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
        for shape in shapes:
            print(shape.shape_type, shape.x, shape.y)

class Shape:
    def __init__(self, x, y, shape_type):
        self.x = x
        self.y = y
        self.shape_type = shape_type
        
    
    
    