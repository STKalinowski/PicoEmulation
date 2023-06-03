import tkinter as tk
from PIL import Image, ImageTk

def display_image(image_path):
    root = tk.Tk()

    # Create a canvas
    canvas = tk.Canvas(root, width=500, height=500)
    canvas.pack()

    try:
        # Open the image
        image = Image.open(image_path)

        # Resize the image to fit the canvas
        image = image.resize((500, 500), Image.ANTIALIAS)

        # Convert the image to Tkinter-compatible format
        tk_image = ImageTk.PhotoImage(image)

        # Display the image on the canvas
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)

        # Run the main loop
        root.mainloop()
    except IOError:
        print("Unable to load image")


# Example usage
image_path = "./TestImg.JPG"  # Replace with the actual image path
display_image(image_path)
