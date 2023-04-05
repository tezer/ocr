import logging

import numpy as np
import numpy.typing as npt
from PIL import Image
from scipy.ndimage import interpolation as inter

import cv2
import pytesseract
from image_data_processor import get_confidence_score, replacements

logger = logging.getLogger("__main__")


def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1])**2)
    return hist, score


def unskew(image: npt.ArrayLike) -> npt.ArrayLike:
    """
    Detects if the image is skewed and rotates it to fix the skew
    :param image:
    :return:
    """
    logger.info("Detecting if the image is skewed.")
    orig_image = image.copy()
    image = Image.fromarray(image)
    delta = .1
    limit = 1
    angles = np.arange(-limit, limit + delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(image, angle)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    if best_angle < 0.1:
        logger.info("The image is not skewed")
        return orig_image
    else:
        logger.info(f"The image is skewed and will be rotated by {best_angle}")
    # correct skew
    data = inter.rotate(image, best_angle, reshape=False, order=0)
    img = Image.fromarray(data)
    return np.asarray(img)


def preprocess(image: npt.ArrayLike) -> npt.ArrayLike:
    """
    Basic filtering
    :param image:
    :return:
    """
    image = unskew(image)
    gray = 255 - cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # prepare a mask using Otsu threshold, then copy from original. this removes some noise
    __, bw = cv2.threshold(cv2.dilate(gray, None), 128, 255, cv2.THRESH_BINARY
                           or cv2.THRESH_OTSU)
    gray = cv2.bitwise_and(gray, bw)
    return gray


def process_image(image_: npt.ArrayLike, language: str) -> str:
    """
    performs iterative cleaning of the image. Tracks image quality metrics (number of recognized words,
    well recognized words to poorly recognized words ratio) to stop the iterations
    :param image_:
    :return: str text produced by tesseract after recognizing the best variant of the preprocessed image
    """
    gray_ = preprocess(image_)
    best_score = 0
    best_balance = .0
    best_step = 0
    processed_images = dict()
    n = 3
    for step in range(120, 0, -10):
        if n <= 0 or best_balance > 10:
            logger.debug(f"Stopped at step {step}")
            break
        logger.info("Trying to improve the original image")
        gray = gray_.copy()
        image = image_.copy()
        processed_image = _process(gray, image, step)
        processed_images[step] = processed_image
        data: str = pytesseract.image_to_data(processed_image)
        good, bad = get_confidence_score(data, step, language)
        score = good
        if score > best_score:
            logger.info("Made some improvement")
            best_score = score  # selecting the best OCR iteration
            best_step = step
            best_balance = float(good) / float(
                bad)  # defining how difficult the OCR was
        elif best_balance > 2:
            n -= 1
        logger.debug(f"{step}: {good}, {bad}")
    logger.debug(
        f"Best step is {best_step} with balance = {best_balance} and score = {best_score}"
    )
    logger.info("Finished image improvement")
    step_replacements = replacements[best_step]
    processed_image = processed_images[best_step]
    string = pytesseract.image_to_string(
        processed_image)  # FIXME need to find a better way of getting the text
    for r in step_replacements:
        if best_balance < 2 or int(r[2]) < 25:
            string = string.replace(r[0], r[1])
            logger.debug(f"Replacing {r[0]} {r[1]} {r[2]}")
    return string


def _process(gray: npt.ArrayLike, image: npt.ArrayLike,
             step: npt.ArrayLike) -> npt.ArrayLike:
    """
    Deletes horizontal lines iterating through different length of the possible lines
    :param gray: grayscale version of the image
    :param image: the image
    :param step: expected length of the line to be removed
    :return: cleaned image (without horizontal lines)
    """
    # make copy of the low-noise underlined image
    gray_copy = gray.copy()
    # scan each row and remove lines
    for row in range(gray.shape[0]):
        for n in range(0, gray.shape[1], step):
            m = n + step
            if m >= len(gray[1]):
                m = len(gray[1])
            avg = np.average(gray[row, n:m] > 26)
            if avg > 0.99:
                cv2.line(image, (n, row), (m, row), (0, 0, 255))
                cv2.line(gray, (n, row), (m, row), (0, 0, 0), 1)
    cont = gray.copy()
    gray_copy1 = gray.copy()
    # after contour processing, the residual will contain small contours
    residual = gray.copy()
    # find contours
    contours, hierarchy = cv2.findContours(cont, cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        # find the boundingbox of the contour
        x, y, w, h = cv2.boundingRect(contours[i])
        if 10 < h:
            cv2.drawContours(image, contours, i, (0, 255, 0), -1)
            # if boundingbox height is higher than threshold, remove the contour from residual image
            cv2.drawContours(residual, contours, i, (0, 0, 0), -1)
        else:
            cv2.drawContours(image, contours, i, (255, 0, 0), -1)
            # if boundingbox height is less than or equal to threshold, remove the contour gray image
            cv2.drawContours(gray, contours, i, (0, 0, 0), -1)
    # now the residual only contains small contours. open it to remove thin lines
    st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, st, iterations=1)
    # prepare a mask for residual components
    __, residual = cv2.threshold(residual, 0, 255, cv2.THRESH_BINARY)

    # combine the residuals. we still need to link the residuals
    combined = cv2.bitwise_or(cv2.bitwise_and(gray_copy1, residual), gray)
    # link the residuals
    st = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 7))
    linked = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, st, iterations=1)

    # prepare a mask from linked image
    __, mask = cv2.threshold(linked, 0, 255, cv2.THRESH_BINARY)
    # copy region from low-noise underlined image
    clean = 255 - cv2.bitwise_and(gray_copy, mask)
    return clean
