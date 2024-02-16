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


def splitMatchedKeypoints(keypoints1, keypoints2, goodMatches):
    points1 = np.float32(
        [keypoints1[match.queryIdx].pt for match in goodMatches])
    points2 = np.float32(
        [keypoints2[match.trainIdx].pt for match in goodMatches])

    return points1, points2


def getFundamentalMatrix(splitKeypoints1, splitKeypoints2, goodMatches):
    fundamentalMatrix, mask = cv2.findFundamentalMat(
        splitKeypoints1, splitKeypoints2, cv2.FM_RANSAC)

    mask = mask.flatten()
    filteredMatches = [goodMatches[i]
                       for i in range(len(goodMatches)) if mask[i]]
    numInliers = len(filteredMatches)
    percentInliers = len(filteredMatches) / len(goodMatches)
    print("Num inliers:", numInliers)
    print("Percent inliers:", percentInliers)

    indecies1 = np.array([match.queryIdx for match in filteredMatches])
    indecies2 = np.array([match.trainIdx for match in filteredMatches])
    xy1 = np.asarray([keypoints1[i].pt for i in indecies1])
    xy2 = np.asarray([keypoints2[i].pt for i in indecies2])
    homogeneous1 = np.hstack(
        (xy1, np.ones((xy1.shape[0], 1))))
    homogeneous2 = np.hstack(
        (xy2, np.ones((xy2.shape[0], 1))))

    outImage = cv2.drawMatches(
        img1, keypoints1, img2, keypoints2, filteredMatches, None)
    cv2.imshow('Passed fundamental inlier test', outImage)
    cv2.waitKey(0)
    return fundamentalMatrix, homogeneous1, homogeneous2, filteredMatches, numInliers, percentInliers


def drawEpipolarLines(img, epipolarLines, keypoints):
    cv2.drawKeypoints(outImage, keypoints, outImage)
    height, width = img.shape[:2]
    for line in epipolarLines:
        a, b, c = line
        # points (x1, y1) and (x2, y2) lie on both the epipolar line and the image border
        if b != 0:  # Avoid division by zero
            x1, x2 = 0, width
            y1 = int(-c / b)
            y2 = int(-(a * width + c) / b)
        else:
            # Line is vertical
            y1, y2 = 0, height
            x1 = int(-c / a)
            x2 = int(-c / a)
        # Generate a random color for each line
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        cv2.line(img, (x1, y1), (x2, y2), color, 2)

    cv2.imshow('Epipolar Lines', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def FundamentalInliers(numInliers, percentageInliers):
    sameScene = numInliers > 25 and percentageInliers > .8
    if sameScene:
        print("almost definitely the same scene\n")
    else:
        print("almost definitely not the same scene\n")
    return sameScene


def getHomography(matches, keypoints1, keypoints2):
    print("Getting homography")
    img1P = np.empty((len(matches), 2), dtype=np.float32)
    img2P = np.empty((len(matches), 2), dtype=np.float32)
    for i in range(len(matches)):
        # -- Get the keypoints from the good matches
        img1P[i, 0] = keypoints1[matches[i].queryIdx].pt[0]
        img1P[i, 1] = keypoints1[matches[i].queryIdx].pt[1]
        img2P[i, 0] = keypoints2[matches[i].trainIdx].pt[0]
        img2P[i, 1] = keypoints2[matches[i].trainIdx].pt[1]

    homography, mask = cv2.findHomography(img1P, img2P, cv2.RANSAC)
    filteredMatches = [matches[i]
                       for i in range(len(matches)) if mask[i]]

    numInliers = len(filteredMatches)
    percentInliers = len(filteredMatches) / len(matches)
    print("Num inliers:", numInliers)
    print("Percent inliers:", percentInliers)

    outImage = cv2.drawMatches(
        img1, keypoints1, img2, keypoints2, filteredMatches, None)
    cv2.imshow('Passed homagraphy inlier test', outImage)
    cv2.waitKey(0)

    return homography, filteredMatches, numInliers, percentInliers


def homographyInliers(numInliers, percentageInliers):
    sameScene = numInliers > 100 and percentageInliers > .5
    if sameScene:
        print("there are more than 200 inliers and the percent of outliers that carried over between steps is greater than 50. \ncalculating mosaic..\n")
    else:
        print("not close enough to calculate mosaic\n")
    return sameScene


def createMosaic(img1, img2, homography):
    result = cv2.warpPerspective(
        img1, homography, (img2.shape[1], img2.shape[0]))

    # Blending the warped image with the second image using alpha blending
    alpha = 0.5  # blending factor
    blended_image = cv2.addWeighted(result, alpha, img2, 1 - alpha, 0)

    # Display the blended image
    cv2.imshow('Blended Image', blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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

            # Creating fundamental matrix
            splitKeypoints1, splitKeypoints2 = splitMatchedKeypoints(
                np.copy(keypoints1), np.copy(keypoints2), goodMatches)
            fundamentalMatrix, homogeneous1, homogeneous2, fundamentalMatches, numInliers, percentInliers = getFundamentalMatrix(
                splitKeypoints1, splitKeypoints2, goodMatches)

            # Draw epipolar lines
            outImage = np.copy(img2)
            epipolarLines = np.dot(fundamentalMatrix, homogeneous1.T).T
            drawEpipolarLines(outImage, epipolarLines, keypoints2)

            sameScene = FundamentalInliers(numInliers, percentInliers)
            if not sameScene:
                continue

            homography, homographyMatches, numInliers, percentInliers = getHomography(
                goodMatches, keypoints1, keypoints2)

            attemptMosaic = homographyInliers(numInliers, percentInliers)
            if not attemptMosaic:
                continue
            createMosaic(img1, img2, homography)

            exit()
