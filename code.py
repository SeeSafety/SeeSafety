
import face_recognition #https://pypi.org/project/face-recognition/

from cv2 import cv2 #https://pypi.org/project/opencv-python/
import numpy as np #https://pypi.org/project/numpy/
import os,math,time

    
import busio #https://pypi.org/project/Adafruit-Blinka/
import board #https://pypi.org/project/board/
    
from gpiozero import Button
import pygame #https://pypi.org/project/pygame/
from scipy.interpolate import griddata #https://pypi.org/project/scipy/
    
from colour import Color #https://pypi.org/project/colour/
    
import adafruit_amg88xx #https://pypi.org/project/Adafruit_AMG88xx/


button_count=0 #amount of times button has been pressed

def thermal(button2): #function that controls thermal camera 

     
    i2ca = busio.I2C(board.SCL, board.SDA) #initiate i2c bus
    
    
    MINIMUM_TEMP,MAXIMUM_TEMP = 26.,32. #range for temperature, corresponds to color spectrum, blue is minimum red is maximum
    
    
    
    
    COLORDEPTH = 1024 #range of possible color values
    
    os.putenv('SDL_FBDEV', '/dev/fb1')
    pygame.init() #initiate pygame library 
    
    
    i2c_data = adafruit_amg88xx.AMG88XX(i2ca) #Initiate the amg8833 module through i2c bus 
    
    
    data_vals= [(math.floor(ix / 8), (ix % 8)) for ix in range(0, 64)] #allows possible combination (0,7)
    grid_x, grid_y = np.mgrid[0:7:32j, 0:7:32j]

    
    
    HEIGHT,WIDTH = 320,480 #size of window(size of lcd)
    
    
    
    INDIGO = Color("indigo") #starting color

    #range from minimum_temp(indigo) to the color red, and COLORDEPTH amount of colors between
    colors_list = list(INDIGO.range_to(Color("red"), COLORDEPTH)) 
    
    
    colors_list = [(int(c.red * 255), int(c.green * 255), int(c.INDIGO * 255)) for c in colors_list] #make an array storing all colors
    
    pixel_width,pixel_height = WIDTH / 20,HEIGHT / 20 #width and height for each set of pixels
    
    window = pygame.display.set_mode((WIDTH, HEIGHT)) #initiate pygame window
    
    window.fill((255, 0, 0)) #fill the screen with all red to start (rgb)
    
    pygame.display.update() #update display 
    pygame.mouse.set_visible(False) #remove mouse 
    
    window.fill((0, 0, 0)) #fill with white(rgb)
    pygame.display.update() #update
    
    
    def constrain(val, mini, maxi): #utility functions that allows value to be constrained 
        return min(maxi, max(mini, val))
    
    def map_value(x, minimum_in, maximum_in, minimum_out, maximum_out): #function that maps value to get resulting color
        return (x - minimum_in) * (maximum_out - minimum_out) / (maximum_in - minimum_in) + minimum_out
    
    
    time.sleep(.1) #give time for sensor to start up 
    
    while True:
        if button2.is_pressed: #quit the loop if the button is presed to switch to facial recognition 
            pygame.quit()
            break
        
        pixel_data = [] #an array storing the other pixels 
        for row in i2c_data.pixel_data: #read all of the pixel data 
            pixel_data = pixel_data + row
        pixel_data = [map_value(p, MINIMUM_TEMP, MAXIMUM_TEMP, 0, COLORDEPTH - 1) for p in pixel_data] 
    
        
        interbicubic = griddata(data_vals, pixel_data, (grid_x, grid_y), method='cubic') #interpole 
    
        #draw the pixels to the screen to show colors for temperature 
        for ix, row in enumerate(interbicubic):
            for jx, pixel in enumerate(row):
                pygame.draw.rect(window, colors_list[constrain(int(pixel), 0, COLORDEPTH- 1)],
                                (pixel_height * ix, pixel_width * jx,
                                pixel_height, pixel_width))
    
        pygame.display.update() #update display



def face_recognitions(button2): #functions that controls facial_recogniton

    video = cv2.VideoCapture(0) #initiate camera and video 
    # use an image that it can use as a refrence to compare to 
    refrence_image = face_recognition.load_image_file("/home/pi/Desktop/eric.jpg")
    refrence_image_encoding = face_recognition.face_encodings(refrence_image)[0]


    recognized_faces = [
        refrence_image_encoding,

    ]
    #names to compare reoognized faces to 
    recognized_faces_names = [
        "Eric",
    ]

    # create arrays to store some data (locations where face was found,names,encoding)
    recognized_face_locations = []
    face_encodings = []
    recognized_names = []
    process = True

    while True:
        if button2.is_pressed: #if a button is pressed break loop to go to thermal camera 
            break
        
        ret, frame = video.read() #select singular frames of the video 

        
        singular_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25) #resize by 0.25 for faster recognition processing speeds


        #a way to convert form opencv's BGR color spectrm to what the face recognition module uses(rgb)
        rgb_singular_frame =singular_frame[:, :, ::-1] 

        #a way to save time by looking at every other frame 
        if process:
            #search for all faces on screen
            recognized_face_locations = face_recognition.recognized_face_locations(rgb_singular_frame)
            face_encodings = face_recognition.face_encodings(rgb_singular_frame, recognized_face_locations)

            recognized_names = []
            for face_encoding in face_encodings:
                
                found_face = face_recognition.compare_faces(recognized_faces, face_encoding) #check for face matches
                name = "Unknown" #for unkown faces 


                #for found faces
                recognized_face_dist = face_recognition.face_distance(recognized_faces, face_encoding)
                best_recognized = np.argmin(recognized_face_dist)
                if found_face[best_recognized]:
                    name = recognized_faces_names[best_recognized]

                recognized_names.append(name)

        process = not process


        # show found faces or unkown faces
        for (top, right, bottom, left), name in zip(recognized_face_locations, recognized_names):
            

            #due to resizing to 0.25, multiply by 4 to get true size 
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2) #draw a rectangle around face to show

            
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED) #make a label under face for name
            font = cv2.FONT_HERSHEY_DUPLEX #font for text
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        
        cv2.imshow('Video', frame) #show the final image 

        # Hit 'q' on the keyboard to quit!
        #if cv2.waitKey(1) & 0xFF == ord('q'):
         #   break

    #ends face recognition window
    video.release() 
    cv2.destroyAllWindows()







button2=Button(21) #initiate button that is connected to pin 21 on raspberry pi 

#way to switch between thermal camera and face recognition by pressing a button 
while True: 
    if button2.is_pressed and button_count%2==0: #if the button is pressed and the amount of times it has been pressed is even 
        button_count+=1
        print(button_count)
        thermal(button2) #thermal camera 
    if button_count%2==1: #if the button is pressed and the amount of times it has been pressed is odd
        button_count+=1
        print(button_count)
        face_recognitions(button2) #facial recognition 
