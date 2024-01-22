import cv2
import numpy as np
import glob
import random
import os

# Load Yolo
net = cv2.dnn.readNet("yolov3_training_last_2.weights", "yolov3_testing.cfg")

# Name custom object
classes = ["Currency"]

# Images path
images_path = glob.glob(r"F:\Lessons\Sem 7\Research\dataset\all Dataset\Currency Image Data Collect for Reseach  (File responses)-20240106T071102Z-001\Currency Image Data Collect for Reseach  (File responses)\50 note second side (File responses)\*g")

output_folder = r'C:\Users\Heshan\Downloads\SHAREit\MI 8 Lite\photo\50_2'
erro_folder=r'C:\Users\Heshan\Downloads\SHAREit\MI 8 Lite\photo\50_2\error'

# os.makedirs(output_folder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not os.path.exists(erro_folder):
    os.makedirs(erro_folder)


layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Insert here the path of your images
# random.shuffle(images_path)

def remove_image(image_path):
    try:
        os.remove(image_path)
        print(f"Image {image_path} removed successfully.")
    except FileNotFoundError:
        print(f"Image {image_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# loop through all the images
img_count = 500
for img_path in images_path:
    # Loading image
    print(img_path)
    img = cv2.imread(img_path)
    img2= cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.85:
                # Object detected
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.75, 0.6)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]

            # Crop the image using bounding box coordinates
            cropped_img = img[y:y + h, x:x + w]
            print(y, h,x , w)

            # Construct the filename with label and index
            filename = str(img_count)+'.jpg'

            # Construct the full save path including the folder
            save_path = os.path.join(output_folder,filename)
            erro_path= os.path.join(erro_folder,filename)

            # Make sure the folder path exists
            os.makedirs(output_folder, exist_ok=True)



            if cropped_img.size != 0:
                cv2.imwrite(save_path, cropped_img)
                print(f"Saved {label} at {save_path}")
                image_to_remove = img_path
                # cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                # cv2.putText(img, label, (x, y + 10), font, 3, color, 2)
                remove_image(image_to_remove)

            else:
                print(f"Error: Cropped image is empty for {label}")
                cv2.imwrite(erro_path, img2)
                image_to_remove = img_path
                # cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                # cv2.putText(img, label, (x, y + 30), font, 3, color, 2)
                remove_image(image_to_remove)

            # cv2.imwrite(save_path, cropped_img)
            # print(f"Saved {label} at {save_path}")

            # F:\Lessons\Sem 7\Research\Operations\Custom model building\cropped images





    # cv2.imshow("Image", img)
    # key = cv2.waitKey(0)
    img_count+=1
cv2.destroyAllWindows()