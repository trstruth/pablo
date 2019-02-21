from pablo import Canvas
from PIL import Image
import numpy as np

c = Canvas('pablo.png')

for i in range(100, 103):
    # open each emoji
    emoji = Image.open('{}/{}.png'.format('/home/trstruth/Projects/pablo/images/emojis', i))
    # emoji.show()
    emoji_arr = np.array(emoji)

    

    '''
    a = np.average(emoji_arr, axis=(0,1), weights=emoji_arr[:][3] != 0)
    print(a)
    '''
    

    '''
    for row in emoji_arr:
        for col in row:
            if col[3] != 0:
                col[3] = 0
            else:
                col[3] = 255
    '''



    # Image.fromarray(emoji_arr).show()
