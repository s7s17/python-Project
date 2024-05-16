import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import numpy as np


class ImageFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Task")
        self.root.geometry("1600x700")

        # Initialize original image
        self.original_image = cv2.imread("1.jpg")
        self.display_image = self.original_image.copy()  # Keep a copy for display

        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.grid(row=0, column=0)

        # Add filter buttons with sliders
        self.add_filter_widgets()

        # Add label for image display
        self.image_label = ttk.Label(self.main_frame)
        self.image_label.grid(row=5, column=0, columnspan=3, padx=20, pady=20)
        self.update_image(self.display_image)

    def add_filter_widgets(self):
        # High-pass filter button and slider
        self.hpf_kernel_size = tk.IntVar(value=5)
        hpf_button = ttk.Button(
            self.main_frame, text="Apply HPF", command=self.apply_hpf
        )
        hpf_button.grid(row=0, column=0, padx=20, pady=20)
        hpf_slider = ttk.Scale(
            self.main_frame,
            from_=1,
            to=20,
            orient=tk.HORIZONTAL,
            variable=self.hpf_kernel_size,
        )
        hpf_slider.grid(row=0, column=1, padx=5, pady=5)

        # Mean filter button and slider
        self.mean_kernel_size = tk.IntVar(value=5)
        mean_button = ttk.Button(
            self.main_frame, text="Apply Mean Filter", command=self.apply_mean_filter
        )
        mean_button.grid(row=0, column=2, padx=5, pady=5)
        mean_slider = ttk.Scale(
            self.main_frame,
            from_=1,
            to=20,
            orient=tk.HORIZONTAL,
            variable=self.mean_kernel_size,
        )
        mean_slider.grid(row=0, column=3, padx=5, pady=5)

        # Median filter button and slider
        self.median_kernel_size = tk.IntVar(value=3)
        median_button = ttk.Button(
            self.main_frame,
            text="Apply Median Filter",
            command=self.apply_median_filter,
        )
        median_button.grid(row=0, column=4, padx=5, pady=5)
        median_slider = ttk.Scale(
            self.main_frame,
            from_=3,
            to=21,
            orient=tk.HORIZONTAL,
            variable=self.median_kernel_size,
        )
        median_slider.grid(row=0, column=5, padx=5, pady=5)

        # Erosion button and slider
        self.erosion_kernel_size = tk.IntVar(value=5)
        erosion_button = ttk.Button(
            self.main_frame, text="Apply Erosion", command=self.apply_erosion
        )
        erosion_button.grid(row=1, column=0, padx=5, pady=5)
        erosion_slider = ttk.Scale(
            self.main_frame,
            from_=1,
            to=20,
            orient=tk.HORIZONTAL,
            variable=self.erosion_kernel_size,
        )
        erosion_slider.grid(row=1, column=1, padx=5, pady=5)

        # Dilation button and slider
        self.dilation_kernel_size = tk.IntVar(value=5)
        dilation_button = ttk.Button(
            self.main_frame, text="Apply Dilation", command=self.apply_dilation
        )
        dilation_button.grid(row=1, column=2, padx=5, pady=5)
        dilation_slider = ttk.Scale(
            self.main_frame,
            from_=1,
            to=20,
            orient=tk.HORIZONTAL,
            variable=self.dilation_kernel_size,
        )
        dilation_slider.grid(row=1, column=3, padx=5, pady=5)

        # Opening button and slider
        self.opening_kernel_size = tk.IntVar(value=5)
        opening_button = ttk.Button(
            self.main_frame, text="Apply Opening", command=self.apply_opening
        )
        opening_button.grid(row=1, column=4, padx=5, pady=5)
        opening_slider = ttk.Scale(
            self.main_frame,
            from_=1,
            to=20,
            orient=tk.HORIZONTAL,
            variable=self.opening_kernel_size,
        )
        opening_slider.grid(row=1, column=5, padx=5, pady=5)

        # Closing button and slider
        self.closing_kernel_size = tk.IntVar(value=5)
        closing_button = ttk.Button(
            self.main_frame, text="Apply Closing", command=self.apply_closing
        )
        closing_button.grid(row=2, column=0, padx=5, pady=5)
        closing_slider = ttk.Scale(
            self.main_frame,
            from_=1,
            to=20,
            orient=tk.HORIZONTAL,
            variable=self.closing_kernel_size,
        )
        closing_slider.grid(row=2, column=1, padx=5, pady=5)

    def apply_hpf(self):
        kernel_size = self.hpf_kernel_size.get()

        # Ensure kernel size is positive and odd
        if kernel_size <= 0 or kernel_size % 2 == 0:
            kernel_size = 5  # Default to an odd kernel size
            self.hpf_kernel_size.set(kernel_size)  # Update the slider value

        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
        # Update displayed image
        self.update_image(blurred_image)

    def apply_mean_filter(self):
        kernel_size = self.mean_kernel_size.get()
        mean_image = cv2.blur(self.original_image, (kernel_size, kernel_size))
        self.update_image(mean_image)

    def apply_median_filter(self):
        kernel_size = self.median_kernel_size.get()
        if kernel_size % 2 == 0:
            kernel_size += 1
        median_image = cv2.medianBlur(self.original_image, kernel_size)
        self.update_image(median_image)

    def apply_erosion(self):
        kernel_size = self.erosion_kernel_size.get()
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        erosion_image = cv2.erode(self.original_image, kernel, iterations=1)
        self.update_image(erosion_image)

    def apply_dilation(self):
        kernel_size = self.dilation_kernel_size.get()
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilation_image = cv2.dilate(self.original_image, kernel, iterations=1)
        self.update_image(dilation_image)

    def apply_opening(self):
        kernel_size = self.opening_kernel_size.get()
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        open_image = cv2.morphologyEx(self.original_image, cv2.MORPH_OPEN, kernel)
        self.update_image(open_image)

    def apply_closing(self):
        kernel_size = self.closing_kernel_size.get()
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        close_image = cv2.morphologyEx(self.original_image, cv2.MORPH_CLOSE, kernel)
        self.update_image(close_image)

    def update_image(self, image):
        # Update the display image and resize if needed to fit the label
        self.display_image = image.copy()  # Keep a copy for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image_rgb.shape
        max_height = 500  # Maximum height for display
        if h > max_height:
            # Resize image while maintaining aspect ratio
            ratio = max_height / h
            image_rgb = cv2.resize(image_rgb, (int(w * ratio), max_height))
        photo = ImageTk.PhotoImage(image=Image.fromarray(image_rgb))
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
