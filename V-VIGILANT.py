#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import pytesseract


plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


cap = cv2.VideoCapture(0)
lst=[]
while True:
    
    ret, frame = cap.read()

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   
    plates = plate_cascade.detectMultiScale(gray, 1.1, 5)

    
    for (x, y, w, h) in plates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi = gray[y:y + h, x:x + w]
        text = pytesseract.image_to_string(roi, lang="eng", config="--psm 7")
        if(len(text)==10):
            lst.append(text)
        
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    
    cv2.imshow("License Plate Detector", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


# In[ ]:




