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
    
    def full_img_fill(self, filename, filename_to_compare):
        img = cv2.imread(filename) 
        img_template = cv2.imread(filename_to_compare) 
        _, im_th = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
        im_floodfill = im_th.copy()
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0,0), (255,255,255))
        sim = self.similarity(im_floodfill, img_template)
        if(sim > 94):
            return (True, sim)
        return (False, None)
    
    def IsSame2x2(self, image_one_filname, image_two_filname):
        image = cv2.imread(image_one_filname)
        image2 = cv2.imread(image_two_filname)
        imageA_array = np.array(image)
        imageB_array = np.array(image2)
        if(np.array_equal(imageA_array, imageB_array)):
            return (True, 100.0)
        
        sim = self.similarity(image, image2)
        if(sim > 97):
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
            if(similarity > 80):
                valid_methods.append((ROTATE, method[0], similarity))
                
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

            if(similarity > 80):
                valid_methods.append((FLIP, method[0], similarity))
        
        if(len(valid_methods) > 1):    
            valid_methods.sort(key=lambda tup: tup[2], reverse=True)
        return valid_methods
    
    # Testing methods
    
    def generate_rotation2x2(self, rotationAmount, answers, image_for_generation=None, image_object=None):
        image = None
        if(image_for_generation is None):
            image = image_object
        else:
            image = cv2.imread(image_for_generation)

        rotated_image = cv2.rotate(image, rotationAmount)
        generated_answers = []
        for index,answer in enumerate(answers):
            ans_image = cv2.imread(answer)
            similarity = self.similarity(rotated_image, ans_image)
            generated_answers.append(GeneratedAnswer(similarity, ROTATE, answerIndex= index + 1))
        return (rotated_image, generated_answers)
    
    def generate_flip2x2(self, flipDirection, answers, image_for_generation=None, image_object=None):
        image = None
        if(image_for_generation is None):
            image = image_object
        else:
            image = cv2.imread(image_for_generation)

        flip = cv2.flip(image, flipCode=flipDirection)
        generated_answers = []
        for index,answer in enumerate(answers):
            answerImage = cv2.imread(answer)
            similarity = self.similarity(flip, answerImage)
            generated_answers.append(GeneratedAnswer(similarity=similarity, method=FLIP, answerIndex=index + 1))
        return (flip, generated_answers)
    
    def generate_same2x2(self, answers, image_for_generation=None, image_object=None):
        image = None
        if(image_for_generation is None):
            image = image_object
        else:
            image = cv2.imread(image_for_generation)

        image_array = np.array(image)
        generated_answers = []
        for index,answer in enumerate(answers):
                answerImage = cv2.imread(answer)
                image_answer = np.array(answerImage)
                if(np.array_equal(image_answer, image_array)):
                    generated_answers.append(GeneratedAnswer(similarity=100.0, method=SAME, answerIndex=index + 1))
                sim = self.similarity(image, answerImage)
                generated_answers.append(GeneratedAnswer(similarity= sim, answerIndex= index + 1, method=SAME))
        return (image, generated_answers)
    
    def generate_fill2x2(self, answers, image_for_generation=None, image_object=None):
        image = None
        if(image_for_generation is None):
            image = image_object
        else:
            image = cv2.imread(image_for_generation)
 
        _, im_th = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
        im_floodfill = im_th.copy()
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(im_floodfill, mask, (0,0), (255,255,255))
        
        generated_answers = []
        for index,answer in enumerate(answers):
                answerImage = cv2.imread(answer)
                sim = self.similarity(im_floodfill, answerImage)
                generated_answers.append(GeneratedAnswer(similarity= sim, answerIndex= index + 1, method=FILL_WHOLE_IMG))
        return (image, generated_answers)
    
    def generate_images(self, method, answers, image_filename=None, imageObject=None):
        # print('---generating answer images----', method)
        method_type = method[0]
        method_amount = method[1]
        if(method_type == SAME):
            return self.generate_same2x2(answers, image_for_generation=image_filename, image_object=imageObject)
        if(method_type == ROTATE):
            return self.generate_rotation2x2(image_for_generation=image_filename, image_object=imageObject, answers=answers, rotationAmount=method_amount)
        if(method_type == FLIP):
            return self.generate_flip2x2(image_for_generation=image_filename, image_object=imageObject, flipDirection=method_amount, answers=answers)
        if(method_type == FILL_WHOLE_IMG):
            return self.generate_fill2x2(image_for_generation=image_filename, image_object=imageObject, answers=answers)

    def generate_and_test(self, a_b_methods, a_c_methods, answers, imageB, imageC):
        a_b_len = len(a_b_methods)
        a_c_len = len(a_c_methods)
        if(a_b_len > 0 and a_c_len > 0):
            # transform both
            print('---has both row and column transformation methods----')
            ans_row_list = []
            # row
            for method in a_b_methods:
                images = self.generate_images(method=method, image_filename=imageC, answers=answers)
                ans_row_list.extend(images[1])
            # column
            ans_column_list = []           
            for method_column in a_c_methods:
                images_column = self.generate_images(method=method_column, image_filename=imageB, answers=answers)
                ans_column_list.extend(images_column[1])

            if(len(ans_column_list) > 1):    
                ans_column_list.sort(key=lambda ans: ans.similarity, reverse=True)
                
            if(len(ans_row_list) > 1):    
                ans_row_list.sort(key=lambda ans: ans.similarity, reverse=True)
                
            if(ans_column_list[0].answerIndex == ans_row_list[0].answerIndex):
                return ans_column_list[0].answerIndex
            
            return ans_row_list[0].answerIndex
        elif(a_b_len > 0):
            # transform image C
            print('---has only row transformation method----')
            ans_list = []
            for method in a_b_methods:
                images = self.generate_images(method=method, image_filename=imageC, answers=answers)
                ans_list.extend(images[1])
            if(len(ans_list) > 1):    
                ans_list.sort(key=lambda ans: ans.similarity, reverse=True)
                for i in ans_list:
                    print(i.method, i.similarity, i.answerIndex)
                return ans_list[0].answerIndex
        
        elif(a_c_len > 0):
            # transform image B
            print('---has only column transformation method----')
            ans_list = []
            for method in a_c_methods:
                images = self.generate_images(method=method, image_filename=imageB, answers=answers)
                ans_list.extend(images[1])

            if(len(ans_list) > 1):    
                ans_list.sort(key=lambda ans: ans.similarity, reverse=True)
                for i in ans_list:
                    print(i.method, i.similarity, i.answerIndex)
                return ans_list[0].answerIndex

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
                A_B_methods.append((SAME, 'h', isSameHorinzontally[1]))
            elif(isSameVertically[0]):
                A_C_methods.append((SAME, 'v', isSameVertically[1]))
            
            # check rotation
            rotation_methodsA_B = self.rotate_check2x2(figures[A].visualFilename, figures[B].visualFilename)
            rotation_methodsA_C = self.rotate_check2x2(figures[A].visualFilename, figures[C].visualFilename)
            A_B_methods.extend(rotation_methodsA_B)
            A_C_methods.extend(rotation_methodsA_C)

            # check flip
            flip_methodsA_B = self.flipAxis2x2(figures[A].visualFilename, figures[B].visualFilename)
            flip_methodsA_C = self.flipAxis2x2(figures[A].visualFilename, figures[C].visualFilename)
            A_B_methods.extend(flip_methodsA_B)
            A_C_methods.extend(flip_methodsA_C)
            
            # check if fill whole image works
            img_fill_row = self.full_img_fill(figures[A].visualFilename, figures[B].visualFilename)
            if(img_fill_row[0]):
                A_B_methods.append((FILL_WHOLE_IMG, 'h', img_fill_row[1]))
            img_fill_col = self.full_img_fill(figures[A].visualFilename, figures[C].visualFilename)
            if(img_fill_col[0]):
                A_C_methods.append((FILL_WHOLE_IMG, 'v', img_fill_col[1]))
                
            # check dark pixel count difference
            
            # Generate and Test
            if(len(A_B_methods) > 1):    
                A_B_methods.sort(key=lambda tup: tup[2], reverse=True)
            if(len(A_C_methods) > 1):    
                A_C_methods.sort(key=lambda tup: tup[2], reverse=True)
            answer_index = self.generate_and_test(A_B_methods, A_C_methods, answers, figures[B].visualFilename, figures[C].visualFilename)
            print('answwe', answer_index)

            return answer_index
        
            
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
        