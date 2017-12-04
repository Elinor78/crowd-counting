import cv2
filenames = ["inputs/{}.mp4".format(x) for x in ["baseball", "london", "many"]]

for filename in filenames:
    print filename
    cap = cv2.VideoCapture(filename)
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            if count == 0:
                f = "outputs/{}_single.jpg".format(filename.split('.')[0].split('/')[1])
                cv2.imwrite(f, frame)
            count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    print count
    cap.release()
    cv2.destroyAllWindows()

    


