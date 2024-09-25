# Optical Mark Recognition (OMR)

## Overview

This project implements Optical Mark Recognition (OMR) to automatically detect and grade multiple-choice answers from scanned answer sheets using OpenCV and Python. The script utilizes various image processing techniques to isolate and recognize marked answers.

## Features

### OMR Process

#### Image Loading
The script loads an image containing an OMR answer sheet.

#### Image Preprocessing
- The image is converted to RGB format and then to grayscale.
- A binary threshold is applied to isolate the answer bubbles.

#### Contour Detection
- Contours are detected in the thresholded image.
- The largest contour (the answer sheet area) is identified.

#### Perspective Transformation
- A perspective transform is applied to obtain a top-down view of the answer sheet.

#### Answer Detection
- Morphological operations are performed to fill gaps in the detected answer bubbles.
- Contours of individual answer bubbles are extracted and filtered based on size.

#### Grading
- Each detected answer is compared against the correct answers to determine the score.
- The grading result is displayed on the output image.

## Output
The result is saved as `result.jpg`, showing the detected answers on the original image, along with the grading percentage.

## Technologies Used

- Python 3.10
- OpenCV
- NumPy
- Matplotlib
