import sys
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


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
    names = []
    for file in os.listdir(directory):
        print(file)
        if file.lower().endswith(".jpg"):
            names.append(os.path.splitext(file)[0])
            filePath = os.path.join(directory, file)
            img = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                print("not none")
                images.append(img)
            else:
                print("Failed to read:", file)
    return images, names


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
    cv2.destroyAllWindows()

    # Decide whether the images depict the same scene, threshholding can be moved
    percent1 = len(goodMatches) / len(keypoints1)
    percent2 = len(goodMatches) / len(keypoints2)
    sameScene = len(goodMatches) > 50 and percent1 > 0.05 and percent2 > 0.05
    sameScene = sameScene or (
        len(goodMatches) > 100 and percent1 > 0.02 and percent2 > 0.02)
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
    cv2.destroyAllWindows()
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
    sameScene = numInliers > 25 and percentageInliers > .6
    sameScene = sameScene or (numInliers > 75 and percentageInliers > .3)
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
    cv2.destroyAllWindows()

    return homography, filteredMatches, numInliers, percentInliers


def homographyInliers(numInliers, percentageInliers):
    if numInliers > 12 and percentageInliers > .5:
        print("there are more than 12 inliers and the percent of outliers that carried over between steps is greater than 50. \ncalculating mosaic..\n")
        return True
    elif numInliers > 40 and percentageInliers > .15:
        print("there are more than 40 inliers and the percent of outliers that carried over between steps is greater than 15. \ncalculating mosaic..\n")
        return True
    else:
        print("not close enough to calculate mosaic\n")
        return False


def createMosaic(img1, img2, homography):
    horizontalBuffer = img2.shape[0] // 2
    verticalBuffer = img2.shape[1] // 2
    img2 = cv2.copyMakeBorder(
        img2, horizontalBuffer, horizontalBuffer, verticalBuffer, verticalBuffer, cv2.BORDER_CONSTANT, value=255)

    translation = np.array([[1, 0, verticalBuffer],
                            [0, 1, horizontalBuffer],
                            [0, 0, 1]])

    homography = np.dot(translation, homography)

    warpedImg1 = cv2.warpPerspective(
        img1, homography, (img2.shape[1], img2.shape[0]))

    alpha = .5
    mosaic = cv2.addWeighted(
        warpedImg1, alpha, img2, 1 - alpha, 0)

    # Display
    cv2.imshow('Mosaic Image', mosaic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def printStats(stats, images):
    os.makedirs(outDir, exist_ok=True)

    # Path to the output file
    outputFile = os.path.join(outDir, 'output.txt')

    # Open the output file for writing
    with open(outputFile, 'w') as file:
        # Determine the maximum width for each column based on data and header length
        max_image_pair_len = max(len(f"{images[i]} & {images[j]}") for i in range(
            len(images)) for j in range(len(images)) if i != j)
        data_max_lengths = [max(len(str(stats[i][j][k])) for i in range(
            len(stats)) for j in range(len(stats[i])) for k in range(4)) for _ in range(4)]

        # Define column headers
        headers = ["Image Pair", "Original Matches",
                   "Num Inliers", "Percent Inliers", "Attempt Mosaic"]

        # Ensure max_lengths accounts for the length of the headers too
        max_lengths = [max(len(headers[0]), max_image_pair_len)] + \
            [max(len(headers[i + 1]), data_max_lengths[i]) for i in range(4)]

        # Adjust header widths based on the maximum lengths
        header_line = ' | '.join(headers[i].ljust(
            max_lengths[i]) for i in range(len(headers)))

        # Print header
        file.write(header_line + "\n")
        file.write('-' * len(header_line))
        file.write("\n")

        # Print table rows
        for i in range(len(stats)):
            for j in range(len(stats[i])):
                if j <= i:  # Optional: to avoid pairing the same image
                    continue
                row_data = [f"{images[i]} & {images[j]}"] + \
                    [str(item) for item in stats[i][j]]
                row_line = ' | '.join(row_data[k].ljust(
                    max_lengths[k]) for k in range(len(row_data))) + "\n"
                file.write(row_line)


if __name__ == "__main__":

    # total arguments
    if (len(sys.argv) != 3):
        print("Wrong number of args\n")
    inDir = sys.argv[1]
    outDir = sys.argv[2]
    images, imageNames = getImages(inDir)
    print("Num images:", len(images))
    stats = np.full((len(images), len(images), 4), [-1, -1, -1, -1])
    for i, img1 in enumerate(images):
        for j, img2 in enumerate(images):
            if (j <= i):
                continue
            # stat tracking variables
            originalMatches = 0
            numInliers = -1
            percentInliers = -1
            attemptMosaic = -1
            stats[i][j] = np.asarray([originalMatches, numInliers,
                                      percentInliers, attemptMosaic])
            keypoints1, descriptors1 = extractKeyPoints(img1)
            keypoints2, descriptors2 = extractKeyPoints(img2)
            goodMatches, sameScene = matchKeyPoints(img1, keypoints1, descriptors1,
                                                    img2, keypoints2, descriptors2)
            originalMatches = len(goodMatches)
            stats[i][j][0] = originalMatches
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

            stats[i][j][1] = numInliers
            stats[i][j][2] = percentInliers * 100

            attemptMosaic = homographyInliers(numInliers, percentInliers)
            stats[i][j][3] = int(attemptMosaic)

            if not attemptMosaic:
                continue
            createMosaic(img1, img2, homography)
    printStats(stats, imageNames)
