import mnist_loader #imports the mnist_loader code
import network #imports network code
import numpy as np #imports numpy for numerical operations
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt #imports numpy for numerical operations
import random

training_data, validation_data, test_data = mnist_loader.load_data_wrapper() #loads and preprocessed the MNIST dataset (inputs reshaped and labels formatted)
training_data = list(training_data) #makes sure that the training data is in a list
test_data = list(test_data) #makes sure that the test data is in a list)

net = network.Network([784, 40, 10]) #creates a neural network with 784 input neurons, 30 neurons in one hidden layer and 10 output neurons (0-9)
net.sgd(training_data, 30, 10, 4, test_data=test_data) #trains the network using stochastic gradient descent into 30 epochs in mini-batches of 10 with a learning rate of 3.0 and evaluates performance on test data after each epoch

def predict_from_array(img_array): #prediction function
    x = img_array.reshape(784, 1) #organizes the 28x28 image in a single column vector
    output = net.feedforward(x) #returns prediction

    print("\nSimilarity percentages:")
    for i, v in enumerate(output): #for loop to show each similarity percentage
        print(f"{i}: {float(v[0])*100:.2f}%")

    print("Prediction:", np.argmax(output)) #prints final prediction


def draw_digit(): #function to draw the number
    width = 200
    height = 200
    window = tk.Tk() #create new window
    window.title("Draw a number between 0 and 9")

    canvas = tk.Canvas(window, width=width, height=height, bg="white")
    canvas.pack()
    image = Image.new("L", (width, height), 255) #create grayscale canvas that you can draw on
    draw = ImageDraw.Draw(image) #draw in the window

    def paint(event):
        x, y = event.x, event.y #x,y coordinates
        r = 8 #pen thickness
        canvas.create_oval(x-r, y-r, x+r, y+r, fill="black") #drawing that we see
        draw.ellipse([x-r, y-r, x+r, y+r], fill=0) #drawing for the computer

    def done():
        img = image.resize((28, 28)) #resize to fit mnist
        img = ImageOps.invert(img) #inverse colors, because the machine reads black background and white numbers
        arr = np.array(img)/255 #set to either 0 or 1 to show black or white for each pixel

        window.destroy()

        predict_from_array(arr)

        plt.imshow(arr, cmap='gray') #show grayscale image of what the computer sees
        plt.title("What computer sees")
        plt.show()

    canvas.bind("<B1-Motion>", paint) #when the left mouse button is down
    button = tk.Button(window, text="Predict", command=done)
    button.pack()

    window.mainloop()

def upload_image(): #upload image option
    path = input("Enter image path: ") #give the image path
    img = Image.open(path).convert("L") #convert to grayscale
    img = img.resize((28, 28))
    img = ImageOps.invert(img)
    arr = np.array(img)/255
    predict_from_array(arr)
    plt.imshow(arr, cmap='gray')
    plt.title("What computer sees")
    plt.show()

def data_set_test():
    z = random.randint(0, 10000)
    x, y = test_data[z] #take random dataset from the 10000 options
    prediction = np.argmax(net.feedforward(x))

    print("\nTest:")
    print("Prediction:", prediction)
    print("Actual:", y)

    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.title(f"Actual: {y}")
    plt.show()

    print("Prediction:", np.argmax(net.feedforward(x)))

while True: #console user interface
    print("\nMNIST Digit Recognizer")
    print("1) Draw a digit")
    print("2) Upload image")
    print("3) Try number from mnist dataset")
    print("4) Exit program")
    choice = input("Choose option: ")
    if choice == "1":
        draw_digit()

    elif choice == "2":
        upload_image()

    elif choice == "3":
        data_set_test()

    elif choice == "4":
        break

    else:
        print("Invalid option")
