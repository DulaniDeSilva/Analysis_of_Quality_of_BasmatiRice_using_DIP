import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk



def load_image():
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if image_path:
        display_image(image_path)
        update_results()

def display_image(image_path):
    image = Image.open(image_path)
    image.thumbnail((400, 400))
    tk_image = ImageTk.PhotoImage(image=image)

    image_label.configure(image=tk_image)
    image_label.image = tk_image

    image_frame.configure(width=image.width(), height=image.height())  # Adapt frame size

def feature_extraction():
    avg_major_axis_var.set("")
    avg_minor_axis_var.set("")
    avg_area_var.set("")
    avg_perimeter_var.set("")

def update_results():
    overall_quality_var.set("Overall Quality: ...")
    overall_chalkiness_var.set("Overall Chalkiness: ...")
    overall_impurity_count_var.set("Overall Impurity Count: ...")

root = tk.Tk()
root.title("Quality Analysis of Basmati Rice")

title_label = tk.Label(root, text="Quality Analysis of Basmati Rice", font=("Helvetica", 16, "bold"))
title_label.pack(pady=10)

image_frame = tk.Frame(root, borderwidth=2, relief="ridge")
image_frame.pack(fill="both", expand=True, padx=20, pady=10)

load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack(pady=10)

image_label = tk.Label(image_frame)
image_label.pack(fill="both", expand=True, padx=20, pady=20)

feature_extraction_frame = tk.LabelFrame(root, text="Feature Extraction")
feature_extraction_frame.pack(anchor="w", padx=20)

avg_major_axis_label = tk.Label(feature_extraction_frame, text="Average Major Axis:")
avg_major_axis_label.grid(row=0, column=0, sticky="w", pady=5)
avg_major_axis_var = tk.StringVar()
avg_major_axis_entry = tk.Entry(feature_extraction_frame, textvariable=avg_major_axis_var)
avg_major_axis_entry.grid(row=0, column=1, sticky="w", padx=10)

avg_minor_axis_label = tk.Label(feature_extraction_frame, text="Average Minor Axis:")
avg_minor_axis_label.grid(row=1, column=0, sticky="w", pady=5)
avg_minor_axis_var = tk.StringVar()
avg_minor_axis_entry = tk.Entry(feature_extraction_frame, textvariable=avg_minor_axis_var)
avg_minor_axis_entry.grid(row=1, column=1, sticky="w", padx=10)

avg_area_label = tk.Label(feature_extraction_frame, text="Average Area:")
avg_area_label.grid(row=2, column=0, sticky="w", pady=5)
avg_area_var = tk.StringVar()
avg_area_entry = tk.Entry(feature_extraction_frame, textvariable=avg_area_var)
avg_area_entry.grid(row=2, column=1, sticky="w", padx=10)

avg_perimeter_label = tk.Label(feature_extraction_frame, text="Average Perimeter:")
avg_perimeter_label.grid(row=3, column=0, sticky="w", pady=5)
avg_perimeter_var = tk.StringVar()
avg_perimeter_entry = tk.Entry(feature_extraction_frame, textvariable=avg_perimeter_var)
avg_perimeter_entry.grid(row=3, column=1, sticky="w", padx=10)

overall_quality_label = tk.Label(root, text="Overall Quality:")
overall_quality_label.pack(anchor="w", padx=20)

overall_quality_var = tk.StringVar()
overall_quality_entry = tk.Entry(root, textvariable=overall_quality_var, state="readonly")
overall_quality_entry.pack(anchor="w", padx=20)

overall_chalkiness_label = tk.Label(root, text="Overall Chalkiness:")
overall_chalkiness_label.pack(anchor="w", padx=20)

overall_chalkiness_var = tk.StringVar()
overall_chalkiness_entry = tk.Entry(root, textvariable=overall_chalkiness_var, state="readonly")
overall_chalkiness_entry.pack(anchor="w", padx=20)

overall_impurity_count_label = tk.Label(root, text="Overall Impurity Count:")
overall_impurity_count_label.pack(anchor="w", padx=20)

overall_impurity_count_var = tk.StringVar()
overall_impurity_count_entry = tk.Entry(root, textvariable=overall_impurity_count_var, state="readonly")
overall_impurity_count_entry.pack(anchor="w", padx=20)

footer_label = tk.Label(root, text="© 2023 Quality Analysis of Basmati Rice")
footer_label.pack(side="bottom", pady=10)

root.mainloop()
