import cv2
import numpy as np
import tkinter as tk


class CircleDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Project - hough transform")
        self.root.geometry('600x600')  # Set the size of the window

        # Load the image directly as a NumPy array
        self.original_image = cv2.imread("2.jpeg")

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.detect_button = tk.Button(
            root, text="Hough Transform", command=self.apply_hough_circle_transform
        )
        self.detect_button.pack()

    def apply_hough_circle_transform(self):
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(
                gray_image,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=50,
                param2=30,
                minRadius=0,
                maxRadius=0,
            )
            if circles is not None:
                circles = np.uint16(np.around(circles))
                hough_image = self.original_image.copy()
                for i in circles[0, :]:
                    cv2.circle(hough_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(hough_image, (i[0], i[1]), 2, (0, 0, 255), 3)
                cv2.imshow("Hough Circles", hough_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == "__main__":
    root = tk.Tk()
    app = CircleDetectionApp(root)
    root.mainloop()
