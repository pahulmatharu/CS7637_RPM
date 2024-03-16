







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

# class ScoringMethod(Enum):
#     DPR = 1
#     IPR = 2

class ImageFrame:
    def __init__(self, image_filename, image_name):
        self.image_name = image_name
        self.image_filename = image_filename
        # denoise image
        image = Utils.denoise_image(image_filename)
        self.image = image
        
        # image metadata
        (ratio_dark_pixels, ratio_white_pixels, num_dark_pixels, num_white_pixel, total_pixels) = Utils.get_pixel_metadata(image)
        self.dark_pixel_count = num_dark_pixels
        self.white_pixel_count = num_white_pixel
        self.white_pixel_percentage = ratio_white_pixels
        self.dark_pixel_percentage = ratio_dark_pixels
        self.total_pixels = total_pixels
        
        # image into quadrants metadata
        # (top_left_dark_pixel_percentage, top_right_dark_pixel_percentage)
        # self.top_left_dark_pixel_percentage 
    
        # create an inverse for contour and connect components
        inv_image = Utils.inverse_image(image)
        self.inv_image = inv_image
        (shapes, shape_count) = Utils.connected_components_analysis(inv_image)
        self.shapes = shapes
        self.shape_count = shape_count
        
        # self.print_metadata()
        
    def print_metadata(self):
        print(self.image_name, self.dark_pixel_count, self.white_pixel_count, self.total_pixels)
        





class Agent:
    def __init__(self):
        self.TwoByTwo = '2x2'
        pass
    
    def similarity(self, image, template):
        gImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply thresholding to convert the image to binary
        thresh_img = cv2.threshold(gImage, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        gTemplate = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        thresh_tmp = cv2.threshold(gTemplate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        return (cv2.matchTemplate(thresh_img, thresh_tmp, cv2.TM_CCOEFF_NORMED).max()) * 100  
    
    def find_same_answer(self, img: ImageFrame, answer_filenames):
        highest_sim = 0
        answer = 1
        for index, filename in enumerate(answer_filenames):
            ans = cv2.imread(filename)
            image = cv2.imread(img.image_filename)
            sim = self.similarity(ans, image)
            if(sim > highest_sim):
                answer = index + 1
                highest_sim = sim
        
        return answer
    
    def is_same_h(self, images):
        for row in images:
            for i in range(0, len(row)-1):
                img1 = row[i]
                img2 = row[i + 1]
                if(img2 is not None and abs(Utils.dark_pixel_ratio(img1, img2)) >= 0.5):
                    return False
        return True
                
    def is_same_v(self, images):
        x = 0
        y = 0
        while(y < len(images[0])):
            while(x < len(images)):
                img1  = images[x][y]
                if(x != 2):
                    img2  = images[x + 1][y]
                    if img2 is not None and abs(Utils.dark_pixel_ratio(img1, img2)) > 0.5:
                            return False
                x = x + 1
                
            y = y + 1
        return True
    
    def is_same_vertical_or_horizontal(self, images: list[list[ImageFrame]]):
        # horizontal
        is_same_h = self.is_same_h(images)
                
        if(is_same_h):
            return images[2][1]
        
        # vertical
        is_same_v = self.is_same_v(images)
        if(is_same_v):
            return images[1][2]
        
        return None
            
    
    def Solve(self, problem):
        name = problem.name
        problemType = problem.problemType
        problemSetName = problem.problemSetName
        hasVisual = problem.hasVisual
        hasVerbal = problem.hasVerbal
        figures = problem.figures

        print('')
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
            
            imageA = ImageFrame(ImageA_filename, A)
            imageB = ImageFrame(ImageB_filename, B)
            imageC = ImageFrame(ImageC_filename, C)
            imageD = ImageFrame(ImageD_filename, D)
            imageE = ImageFrame(ImageE_filename, E)
            imageF = ImageFrame(ImageF_filename, F)
            imageG = ImageFrame(ImageG_filename, G)
            imageH = ImageFrame(ImageH_filename, H)
            
            images = [[imageA, imageB, imageC], [imageD, imageE, imageF], [imageG, imageH, None]]
            answer_filenames = [answerImg_filename1, answerImg_filename2, answerImg_filename3, answerImg_filename4, answerImg_filename5, answerImg_filename6, answerImg_filename7, answerImg_filename8]
            # if any row or column rule is keep image same, then return the image that should be the answer.

            # scoring = {
            #     1: 0,
            #     2: 0,
            #     3: 0,
            #     4: 0,
            #     5: 0,
            #     6: 0,
            #     7: 0,
            #     8: 0
            # }
            # # column
            # print('---column---')
            dpr_ag = Utils.dark_pixel_ratio(imageA, imageG)
            ipr_ag = Utils.intersection_pixel_ratio(imageA, imageG)
            # print('A-G', ipr_ag, dpr_ag)
            
            # dpr_bh = Utils.dark_pixel_ratio(imageB, imageH)
            # ipr_bh = Utils.intersection_pixel_ratio(imageB, imageH)
            # print('B-H', ipr_bh, dpr_bh)
            
            # print('---row---')
            dpr_ac = Utils.dark_pixel_ratio(imageA, imageC)
            ipr_ac = Utils.intersection_pixel_ratio(imageA, imageC)
            # print('A-C', ipr_ac, dpr_ac)
            
            # dpr_df = Utils.dark_pixel_ratio(imageD, imageF)
            # ipr_df = Utils.intersection_pixel_ratio(imageD, imageF)
            # print('D-F', ipr_df, dpr_df)
            
            min_ipr = 100000
            final_answer = 1
            min_dpr = 100000
            for index, ans in enumerate(answer_filenames):
                answer_index = index + 1
                # column
                ans_frame = ImageFrame(ans, answer_index)
                ipr_ans = Utils.intersection_pixel_ratio(imageC, ans_frame)
                dpr_ans = Utils.dark_pixel_ratio(imageC, ans_frame)
                (ipr_distance, dpr_distance) = Utils.ipr_dpr_scoring(ipr_ag, dpr_ag, ipr_ans, dpr_ans)
                # bh_score = Utils.ipr_dpr_scoring(ipr_bh, dpr_bh, ipr_ans, dpr_ans)
                
                # row
                ipr_row_ans = Utils.intersection_pixel_ratio(imageG, ans_frame)
                dpr_row_ans = Utils.dark_pixel_ratio(imageG, ans_frame)
                (ipr_row_distance, dpr_row_distance) = Utils.ipr_dpr_scoring(ipr_ac, dpr_ac, ipr_row_ans, dpr_row_ans)
                # df_score = Utils.ipr_dpr_scoring(ipr_df, dpr_df, ipr_ans, dpr_ans)
                
                # print(f'answer {answer_index} row', f'ipr:{ipr_row_ans}', f'ipr_distance:{ipr_row_distance}', f'dpr_{dpr_row_ans}', f'dpr distance: {dpr_row_distance}')
                # print(f'answer {answer_index} column', f'ipr:{ipr_ans}', f'ipr distance:{ipr_distance}', f'dpr_{dpr_ans}', f'dpr distance: {dpr_distance}')
                
                if(ipr_distance > ipr_row_distance):
                    if(min_ipr > ipr_row_distance):
                        min_ipr = ipr_row_distance
                        final_answer = answer_index
                else:
                    if(min_ipr > ipr_distance):
                        min_ipr = ipr_distance
                        final_answer = answer_index
                
            print(f'------- answer ------- : {final_answer}')
            return final_answer
            # print(Utils.dark_pixel_ratio(imageB, imageC))
            # print(Utils.dark_pixel_ratio(imageD, imageE))
            # print(Utils.dark_pixel_ratio(imageE, imageF))
            # print(Utils.dark_pixel_ratio(imageG, imageH))
            # print(Utils.dark_pixel_ratio(imageD, imageG))
            # print(Utils.dark_pixel_ratio(imageB, imageE))
            # print(Utils.dark_pixel_ratio(imageE, imageH))
            # print(Utils.dark_pixel_ratio(imageC, imageF))
            
            # rules
            potential_answer = self.is_same_vertical_or_horizontal(images)
            if(potential_answer is not None):
                answer = self.find_same_answer(potential_answer, answer_filenames)
                print(answer)
                return answer
            
            # start weighting system
            

        return 1
    
    

class Utils:
    def __init__(self):
        pass 
    
    @staticmethod
    def denoise_image(image_filename):
        image = cv2.imread(image_filename)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
        bg=cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
        out_gray=cv2.divide(image, bg, scale=255)
        return cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1]
    
    @staticmethod
    def ipr_dpr_scoring(template_ipr, template_dpr, answer_ipr, answer_dpr):
        ipr_distance = abs(answer_ipr - template_ipr)
        dpr_distance = abs(answer_dpr - template_dpr)
        
        return (ipr_distance, dpr_distance)
        # if(scoring_method == ScoringMethod.DPR):
        #     return Utils.scoring_distance(dpr_distance)
        # return Utils.scoring_distance(ipr_distance)
        
    @staticmethod
    def scoring_distance(value):
        # if exact, meaning dif < 0.0005,  +4.
        # if exact, meaning dif < 0.001,  +3.
        # if exact, meaning dif = 0.01,  +2.
        # if exact, meaning dif = 0.05,  +1.
        
        points = 0
        if(value == 0.0):
            points = points + 5
        elif(value <= 0.0005):
            points = points + 4
        elif(value <= 0.001):
            points = points + 3
        elif(value <= 0.01):
            points = points + 2
        elif(value <= 0.05):
            points = points + 1
        return points

    @staticmethod
    def dark_pixel_ratio(image: ImageFrame, image2: ImageFrame):
        return (image.dark_pixel_percentage - image2.dark_pixel_percentage)
    
    @staticmethod
    def intersection_pixel_ratio(image_one_not_inversed: ImageFrame, image_two_not_inversed: ImageFrame):
        #intersection of dark pixels
        img_intersection = cv2.bitwise_or(image_one_not_inversed.image, image_two_not_inversed.image)
        # cv2.imshow("or", img_bwo)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        (ratio_dark_pixels, ratio_white_pixels, num_dark_pixels, num_white_pixel, total_pixels) = Utils.get_pixel_metadata(img_intersection)
        # print(ratio_dark_pixels, ratio_white_pixels, num_dark_pixels, num_white_pixel, total_pixels)
        
        return ((ratio_dark_pixels / image_one_not_inversed.dark_pixel_percentage) - (ratio_dark_pixels / image_two_not_inversed.dark_pixel_percentage))
        
        # img_bwa = cv2.bitwise_and(image_one_not_inversed.image, image_two_not_inversed.image)
        # img_bwx = cv2.bitwise_not(cv2.bitwise_xor(image_one_not_inversed.image, image_two_not_inversed.image))

        # cv2.imshow("and", img_bwa)
        # cv2.imshow("xor", img_bwx)
    
    @staticmethod
    def contour(image):
        # # setting threshold of gray image 
        _, threshold_image = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY) 
        
        # # using a findContours() function 
        contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Iterate over the contours
        for contour in contours:

            # Approximate the contour
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        
    @staticmethod
    def connected_components_analysis(image):
        ret, thresh = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # You need to choose 4 or 8 for connectivity type
        connectivity = 4  
        # Perform the operation
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
        
        # Shape count: num_labels variable will contain the total number of shapes in the image, including the background. So subtract 1 
        shape_count = num_labels - 1
        shapes = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            shapes.append(Shape(x, y, w, h, area))
        
        # label_hue = np.uint8(179*labels/np.max(labels))
        # blank_ch = 255*np.ones_like(label_hue)
        # labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        # # cvt to BGR for display
        # labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        # # set bg label to black
        # labeled_img[label_hue==0] = 0
        # cv2.imshow('labeled.png', labeled_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        
        return (shapes, shape_count)
    
    @staticmethod
    def split_into_four_quadrants(image):
        (h, w) = image.shape[:2]
        x_center = w // 2
        y_center = h // 2
        
        top_left_quad = image[0:y_center, 0: x_center]
        top_right_quad = image[0:y_center, x_center:]
        bottom_left_quad = image[y_center:, 0: x_center]
        bottom_right_quad = image[y_center:, x_center:]
        
        # cv2.imshow('top_left_quad', top_left_quad)
        # cv2.imshow('top_right_quad', top_right_quad)
        # cv2.imshow('bottom_left_quad', bottom_left_quad)
        # cv2.imshow('bottom_right_quad', bottom_right_quad)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
    
    @staticmethod
    def get_pixel_metadata(image_not_inversed):
        (x, y) = image_not_inversed.shape[:2]
        num_white_pixel = np.sum(image_not_inversed >= 1) 
        num_dark_pixels = np.sum(image_not_inversed == 0)
        total_pixels = x * y
        ratio_dark_pixels = (num_dark_pixels / total_pixels) * 100
        ratio_white_pixels = (num_white_pixel / total_pixels) * 100
        return (ratio_dark_pixels, ratio_white_pixels, num_dark_pixels, num_white_pixel, total_pixels)
    
    @staticmethod
    def inverse_image(image):
        # because connected components expect white on black, we need to reverse the image
        return cv2.bitwise_not(image)
    
    
class Shape:
    def __init__(self, x,y,w,h,area):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.area = area
