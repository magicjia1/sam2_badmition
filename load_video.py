
import os



import cv2

def getFrame(videoPath, svPath):
    cap = cv2.VideoCapture(videoPath)
    numFrame = 0  # 图片名称的计数起始值
    while True:
        ret, frame = cap.read()  # 使用 read() 替代 grab() 和 retrieve()
        if not ret:
            break  # 如果正确读取帧，ret为True，否则为False，循环结束
        else:
            # 格式化文件名，确保 numFrame 是五位数字
            filename = f"{svPath}{numFrame:05}.jpg"
            # 将帧保存为图片
            cv2.imwrite(filename, frame)  # 使用 imwrite() 替代 imencode() 和 tofile()
        numFrame += 1  # 增加帧计数

    # 释放视频捕获对象
    cap.release()

if __name__ == '__main__':
    videoPath = "./video1/badmotion2.mp4"
    output_dir = './video1/videos/'
    getFrame(videoPath, output_dir)

    image_path = output_dir+"00000.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cv2.imshow(image);