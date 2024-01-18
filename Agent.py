# Allowable libraries:
# - Python 3.10.12
# - Pillow 10.0.0
# - numpy 1.25.2
# - OpenCV 4.6.0 (with opencv-contrib-python-headless 4.6.0.66)

# To activate image processing, uncomment the following imports:
from PIL import Image
import numpy as np
# import cv2

class Agent:
    def __init__(self):
        self.TwoByTwo = '2x2'
        self.A = 'A'
        self.B = 'B'
        self.C = 'C'
        self.D = 'D'
        self.E = 'E'
        self.F = 'F'
        self.G = 'G'
        self.H = 'H'
        """
        The default constructor for your Agent. Make sure to execute any processing necessary before your Agent starts
        solving problems here. Do not add any variables to this signature; they will not be used by main().
        
        This init method is only called once when the Agent is instantiated 
        while the Solve method will be called multiple times. 
        """
        pass

    def IsMirror2x2(self, imageA, imageB):
        imageA_array = np.array(imageA)
        imageB_array = np.array(imageB)
        return np.array_equal(imageA_array, imageB_array)

    def Solve(self, problem):
        """
        Primary method for solving incoming Raven's Progressive Matrices.

        Args:
            problem: The RavensProblem instance.

        Returns:
            int: The answer (1-6 for 2x2 OR 1-8 for 3x3).
            Return a negative number to skip a problem.
            Remember to return the answer [Key], not the name, as the ANSWERS ARE SHUFFLED in Gradescope.
        """

        '''
        DO NOT use absolute file pathing to open files.
        
        Example: Read the 'A' figure from the problem using Pillow
            image_a = Image.open(problem.figures["A"].visualFilename)
            
        Example: Read the '1' figure from the problem using OpenCv
            image_1 = cv2.imread(problem.figures["1"].visualFilename)
            
        Don't forget to uncomment the import!
        '''
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
            QuestionImageA = Image.open(figures[self.A].visualFilename)
            QuestionImageB = Image.open(figures[self.B].visualFilename)
            QuestionImageC = Image.open(figures[self.C].visualFilename)
            # Answers
            AnswerImage1 = Image.open(figures['1'].visualFilename)
            AnswerImage2 = Image.open(figures['2'].visualFilename)
            AnswerImage3 = Image.open(figures['3'].visualFilename)
            AnswerImage4 = Image.open(figures['4'].visualFilename)
            AnswerImage5 = Image.open(figures['5'].visualFilename)
            AnswerImage6 = Image.open(figures['6'].visualFilename)
            answers = [AnswerImage1, AnswerImage2, AnswerImage3, AnswerImage4, AnswerImage5, AnswerImage6]

            # see if the relationship between A and B is a mirror.
            isMirror = self.IsMirror2x2(QuestionImageA, QuestionImageB)
            if(isMirror):
                for index, answer in enumerate(answers):
                    isMirrorToC = self.IsMirror2x2(QuestionImageC, answer)
                    if(isMirrorToC):
                        print(index + 1)
                        return index + 1
        else: 
            # 3x3
            # Question
            ImageA = Image.open(figures[self.A].visualFilename)
            ImageB = Image.open(figures[self.B].visualFilename)
            ImageC = Image.open(figures[self.C].visualFilename)
            ImageD = Image.open(figures[self.D].visualFilename)
            ImageE = Image.open(figures[self.E].visualFilename)
            ImageF = Image.open(figures[self.F].visualFilename)
            ImageG = Image.open(figures[self.G].visualFilename)
            ImageH = Image.open(figures[self.H].visualFilename)

            #Answers
            Image1 = Image.open(figures['1'].visualFilename)
            Image2 = Image.open(figures['2'].visualFilename)
            Image3 = Image.open(figures['3'].visualFilename)
            Image4 = Image.open(figures['4'].visualFilename)
            Image5 = Image.open(figures['5'].visualFilename)
            Image6 = Image.open(figures['6'].visualFilename)
            Image7 = Image.open(figures['7'].visualFilename)
            Image8 = Image.open(figures['8'].visualFilename)

        # Placeholder: Skip all problems for now.
        return 1