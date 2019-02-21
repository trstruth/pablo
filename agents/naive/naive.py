# coding: utf-8

"""Accepts a path to an image and returns it emojified

IMPORTANT PARAMETERS:
size of emojis, number of iterations, various paths towards emojis, from the desired image, and destination directory.
"""  

from PIL import Image
import numpy as np
import os
import copy
import random
from tqdm import tqdm

def sampler(image):
    """Picks out points on image."""
    len_x, len_y = (image.size[0], image.size[1])
    x, y = random.randint(0, len_x-1), random.randint(0, len_y-1)
    return x, y

def getAverageRGB(image):
    """Given PIL Image, return its Average RGBA value."""
    im = np.array(image)
    w,h,d = im.shape
    im.shape = (w*h, d)
    return tuple(im.mean(axis=0))

def getListOfRGBs(path="../images/emojis/"):
    """Given a path towards folder with all N emojis build a long (4*N) table with the average RGBA values of each emoji."""
    list_of_average_RGBS = np.zeros((4, 2363))
    for i in range(2363):
        new_path = path+str(i)+'.png'
        emoji = Image.open(new_path)
        average_rgb = np.array(getAverageRGB(emoji))
        list_of_average_RGBS[:, i] = average_rgb   
    return list_of_average_RGBS

def findEmojiNumber(pixel_value, list_of_average_RGBs, tol = 4000, num_emojis = 2362):
    """Given a pixel value sampled from the image, and a list of the average color of different emojis, 
    find an emoji that approximates the colour of the sampled pixel.

    Approximation is done by comparing the pixel RGB values to those of each emoji until the euclidean 
    distance between the two RGB vectors is below a tolerance threshold `tol`.
    """
    offset = random.randint(0, num_emojis)
    for iters in range(num_emojis):
        i = (iters + offset) % (num_emojis-1)
        distance = np.sum((pixel_value - list_of_average_RGBs[1:4, i])**2)
        if distance < tol:
            return i
    return 0

def findEmojiNumberPrecise(pixel_value, list_of_average_RGBs):
    """Given a pixel value sampled from the image, and a list of the average color of different emojis, 
    find an emoji that approximates the colour of the sampled pixel.

    Approximation is done by finding the minimum distance between pixel RGB value and average emoji RGB value, 
    using the table computed in getListOfRGBs().

    TODO: Add tolerance so that it doesn't pick the min every time, but alternates between the k smallest ones.
    """
    pixel_value = np.array(pixel_value[0:3]).reshape((3,1)) #we don't care about alpha here
    list_of_average_RGBs = list_of_average_RGBs[0:3, :] #we don't care about alpha here
    distance_table = np.sum((pixel_value - list_of_average_RGBs)**2, axis=0)
    return distance_table.argmin()


def pasteEmoji(image, emoji_number, location, emoji_size = 25, path="../images/emojis/"):
    """Paste an emoji with number `emoji_number` onto an `image` at location `location`.

    TODO: Add more modification options, like rotation, resizing, etc.
    """
    new_path = path+str(emoji_number)+".png"
    emoji = Image.open(new_path)
    emoji = emoji.resize((emoji_size,emoji_size))

    image.paste(emoji, (location[0]-int(emoji_size/2), location[1]-int(emoji_size/2)), emoji)


def main():
    """Runs entire script

    ?TODO?: System arguments to run it from terminal
    """
    
    #setup vars
    iters = 100000
    image_name = "pablo" # change accordingly
    base = Image.open(image_name+".png", "r")
    image = base.copy()
    PATH_TO_EMOJI = "pablo/images/emojis/" 
    emoji_size = 25 
    list_of_RGBs = getListOfRGBs()

    for i in tqdm(range(iters)):
        location = sampler(base)
        pixel = base.getpixel(location)
        pixel = pixel[0:3]
        
        emoji_num = findEmojiNumberPrecise(pixel, list_of_RGBs)
        emoji = pasteEmoji(image, emoji_num, location, emoji_size=emoji_size)


    image.save("new_"+image_name+"_"+str(iters)+".png")
    


if __name__ == "__main__":
    main()
