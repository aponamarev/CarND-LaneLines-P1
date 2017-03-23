import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

class Lane(object):

    def __init__(self, height=0.43,
                 canny_lowT = 150,
                 canny_highT = 200,
                 rho = 1,
                 theta = np.pi / 180,
                 min_line_len = 24,
                 threshold = 16,
                 max_line_gap = 6,
                 kernel_size = 3,
                 split = 0.3):

        self.height = height
        self.__verticies = []
        self.__canny_lowT = canny_lowT
        self.__canny_highT = canny_highT
        self.rho = rho
        self.theta = theta
        self.min_line_len = min_line_len
        self.threshold = threshold
        self.max_line_gap = max_line_gap
        self.shapes = [640, 720, 3]
        self.__kernel_size = kernel_size
        self.__split = split

        self.__buffer = [[478, 309, 167, 540],[482, 309, 864, 540]]


    @property
    def shapes(self):
        return self.__shapes

    @shapes.setter
    def shapes(self, value):
        self.__shapes = value
        self.__verticies = np.array([[[0, self.__shapes[0]],
                                      [int(self.__shapes[1] / 2), int(self.__shapes[0] * (1-self.height))],
                                      [self.__shapes[1], self.__shapes[0]]]])


    def __grayscale(self, img):
        """Applies the Grayscale transform
        This will return an image with only one color channel
        but NOTE: to see the returned image as grayscale
        (assuming your grayscaled image is called 'gray')
        you should call plt.imshow(gray, cmap='gray')"""
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def __norm_img(self, img):
        shapes = img.shape
        img = np.array(img, dtype=np.float32)
        channels = 1 if len(shapes) < 3 else shapes[-1]
        flat = np.reshape(img, (-1, channels))
        avg = np.mean(flat)
        flat -= avg
        std = np.std(flat)
        flat = flat / std
        flat -= flat.min()
        range = flat.max() - flat.min()
        flat = flat * (255 / range)
        flat = np.array(flat, dtype=np.uint8)
        return np.reshape(flat, shapes)

    def __canny(self, img):
        """Applies the Canny transform"""
        return cv2.Canny(img, self.__canny_lowT, self.__canny_highT)

    def __gaussian_blur(self, img):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (self.__kernel_size, self.__kernel_size), 0)

    def __region_of_interest(self, img):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, self.__verticies, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def __hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        """
        `img` should be the output of a Canny transform.

        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                maxLineGap=max_line_gap)
        return lines

    def __filter_lines(self, lines):
        ms = []
        for line in lines:
            for x0, y0, x1, y1 in line:
                ms.append((y1 - y0) / (x1 - x0))

        ms = np.array(ms)
        right_set = lines[ms < -self.__split]

        left_set = lines[ms > self.__split]

        result = np.concatenate((right_set, left_set))

        return result

    def __find_2lines(self, lines):

        ms = []

        for line in lines:
            for x0, y0, x1, y1 in line:
                ms.append((x1 - x0) / (y1 - y0))

        split = self.__split
        ms = np.array(ms)
        right_set = lines[ms <= split]
        right_set = np.reshape(right_set, (-1, 2))
        right_lane = np.polyfit(right_set[:, 1], right_set[:, 0], 1)

        left_set = lines[ms > split]
        left_set = np.reshape(left_set, (-1, 2))
        left_lane = np.polyfit(left_set[:, 1], left_set[:, 0], 1)

        return right_lane, left_lane

    def __calculate_lines(self, line, shape, portion=0.5):
        return [
            [int(line[1] + line[0] * shape[0] * portion), int(shape[0] * portion), int(line[1] + line[0] * shape[0]),
             int(shape[0])]]


    def __line_hight(self, lines):
        m = 10 ** 6
        for line in lines:
            for x0, y0, x1, y1 in line:
                m = min(m, y0, y1)

        return m

    def __img_preprocessing_pipeline(self, img):

        self.shapes = img.shape

        preprocessed = self.__grayscale(img)
        preprocessed = self.__norm_img(preprocessed)
        preprocessed = self.__canny(preprocessed)
        preprocessed = self.__region_of_interest(preprocessed)

        return preprocessed


    def get_lanes(self, img):

        processed_img = self.__img_preprocessing_pipeline(img)
        #plt.imshow(processed_img)
        lines = self.__hough_lines(processed_img,
                                   self.rho,
                                   self.theta,
                                   self.threshold,
                                   self.min_line_len,
                                   self.max_line_gap)

        try:
            lines = self.__filter_lines(lines)
            height = self.__line_hight(lines)
            right_lane, left_lane = self.__find_2lines(lines)
            right_lane = self.__calculate_lines(right_lane, self.shapes, portion=height/self.shapes[0])
            left_lane = self.__calculate_lines(left_lane, self.shapes, portion=height / self.shapes[0])
            self.__buffer = [right_lane, left_lane]
        except:
            right_lane, left_lane = self.__buffer

        return right_lane, left_lane

    def provide_img_with_lanes(self, img):

        initial_img = img

        right, left = self.get_lanes(img)
        mask = np.zeros_like(img, dtype=np.uint8)
        self.draw_lines(mask, [right, left])

        img_with_lanes = cv2.addWeighted(initial_img, 0.8, mask, 1.0, 0.)

        return img_with_lanes

    def draw_lines(self, img, lines, color=[255, 0, 0], thickness=2):
        """
        NOTE: this is the function you might want to use as a starting point once you want to
        average/extrapolate the line segments you detect to map out the full
        extent of the lane (going from the result shown in raw-lines-example.mp4
        to that shown in P1_example.mp4).

        Think about things like separating line segments by their
        slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
        line vs. the right line.  Then, you can average the position of each of
        the lines and extrapolate to the top and bottom of the lane.

        This function draws `lines` with `color` and `thickness`.
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


if __name__ == "__main__":

    import os

    lane = Lane()

    read_path = "test_images/"
    save_path = "test_images_output/"
    files = os.listdir("test_images/")

    file_name = files[0]
    img_orig = plt.imread(os.path.join(read_path, file_name))

    img_with_lanes = lane.provide_img_with_lanes(img_orig)
    plt.imshow(img_with_lanes, cmap="gray")

    print("done")



