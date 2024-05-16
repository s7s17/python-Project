import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

class ImageFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Priject - Filter App")
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky="ew")
        # Initialize original image
        self.load_image_button = ttk.Button(
            self.main_frame,
            text="Load Image",
            command=self.load_image

        )

        self.load_image_button.grid(row=0, column=0, padx=5, pady=5)
        # Add buttons for filter application
        self.add_buttons()
        # Placeholder for image label
        self.image_label = ttk.Label(self.main_frame)
        self.image_label.grid(row=3, column=0, columnspan=3)

    def load_image(self):
        self.original_image = cv2.imread("1.jpg")
        self.display_image(self.original_image)

    def add_buttons(self):
        # Button to apply Roberts edge detector
        roberts_button = ttk.Button(
            self.main_frame,
            text="Apply Roberts Edge Detector",
            command=self.apply_roberts_edge_detector,
        )
        roberts_button.grid(row=1, column=0, padx=5, pady=5)
        # Button to apply Prewitt edge detector
        prewitt_button = ttk.Button(
            self.main_frame,
            text="Apply Prewitt Edge Detector",
            command=self.apply_prewitt_edge_detector,
        )
        prewitt_button.grid(row=1, column=1, padx=5, pady=5)
        # Button to apply Sobel edge detector
        sobel_button = ttk.Button(
            self.main_frame,
            text="Apply Sobel Edge Detector",
            command=self.apply_sobel_edge_detector,
        )
        sobel_button.grid(row=1, column=2, padx=5, pady=5)

    def apply_roberts_edge_detector(self):
        # Assuming the use of Canny for Roberts is intentional
        roberts_image = cv2.Canny(self.original_image, 100, 200)
        self.display_image(roberts_image)

    def apply_prewitt_edge_detector(self):
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewitt_x = cv2.filter2D(gray_image, -1, kernel_x)
        prewitt_y = cv2.filter2D(gray_image, -1, kernel_y)
        prewitt_image = cv2.bitwise_or(prewitt_x, prewitt_y)
        self.display_image(prewitt_image)

    def apply_sobel_edge_detector(self):
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_image = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel_image = cv2.normalize(
            sobel_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        self.display_image(sobel_image)

    def display_image(self, image):
        # Resize the image
        base_width = 800  # Set the desired width
        img_ratio = image.shape[0] / image.shape[1]
        base_height = int(base_width * img_ratio)
        resized_image = cv2.resize(image, (base_width, base_height), interpolation=cv2.INTER_AREA)

        # Convert to RGB for display
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(image_rgb))

        # Update the image label
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def run(self):
        self.root.mainloop()

def main():
    root = tk.Tk()
    app = ImageFilterApp(root)
    app.run()

if __name__ == "__main__":
    main()
