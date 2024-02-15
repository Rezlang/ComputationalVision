import sys
import cv2
import numpy as np
import os


def getImages(inDir):
    """returns list of numpy images

    Args:
        inDir (string): relative input file directory
    Returns:
        arrayList: list of greyscale images
    """
    print("Getting images")
    images = []
    directory = os.path.join('.', inDir)

    for file in os.listdir(directory):
        print(file)
        if file.lower().endswith(".jpg"):
            print("ends with jpg")
            filePath = os.path.join(directory, file)
            img = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                print("not none")
                images.append(img)
            else:
                print("Failed to read:", file)
    return images


def extractKeyPoints(img):
    """_summary_

    Args:
        img (np.array): image

    Returns:
        arrayList[keypoint]: list of keypoints found in image
        np.array
    """
    print("Extracting key points")
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    tempImg = np.copy(img)
    cv2.drawKeypoints(img, keypoints, tempImg)
    cv2.imshow('SIFT Keypoints', tempImg)
    cv2.waitKey(0)
    print("Num keypoints:", len(keypoints))
    return keypoints, descriptors


def matchKeyPoints(img1, keypoints1, descriptors1, img2, keypoints2, descriptors2):
    print("Matching keypoints")
    matcher = cv2.FlannBasedMatcher()
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
    print("Number of matches before ratio test:", len(knn_matches), "\n")

    # Ratio test
    RATIO_THRESHHOLD = 0.75
    goodMatches = []
    for closestMatch, secondClosest in knn_matches:
        if closestMatch.distance < RATIO_THRESHHOLD * secondClosest.distance:
            goodMatches.append(closestMatch)
    goodMatches = np.asarray(goodMatches)

    # Output text and image
    print("Number of matches:", len(goodMatches), "\n")
    print("Fractional match (img 1): {:.2f}".format(
        len(goodMatches) / len(keypoints1)))
    print("Fractional match (img 2): {:.2f}".format(
        len(goodMatches) / len(keypoints2)))
    outImage = cv2.drawMatches(
        img1, keypoints1, img2, keypoints2, goodMatches, None)
    cv2.imshow('drawnMatches', outImage)
    cv2.waitKey(0)

    # Decide whether the images depict the same scene, threshholding can be moved
    sameScene = len(goodMatches) > 75 and len(
        goodMatches) / len(keypoints1) > 0.1 and len(goodMatches) / len(keypoints2) > 0.1
    if sameScene:
        print("Probably the same scene\n")
    else:
        print("Probably not the same scene\n")

    return goodMatches, sameScene


def getFundamentalMatrix():
    return


if __name__ == "__main__":

    # total arguments
    if (len(sys.argv) != 3):
        print("Wrong number of args\n")
    inDir = sys.argv[1]
    outDir = sys.argv[2]
    images = getImages(inDir)
    print("Num images:", len(images))

    for img1 in images:
        for img2 in images:
            if (img1 == img2).all():
                continue
            keypoints1, descriptors1 = extractKeyPoints(img1)
            keypoints2, descriptors2 = extractKeyPoints(img2)
            goodMatches, sameScene = matchKeyPoints(img1, keypoints1, descriptors1,
                                                    img2, keypoints2, descriptors2)
            if not sameScene:
                continue

            exit()
