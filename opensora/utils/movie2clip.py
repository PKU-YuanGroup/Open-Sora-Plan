import cv2
from tqdm import tqdm


def crop_video(input_file, start_and_end):
    # 打开视频文件
    video = cv2.VideoCapture(input_file)

    # 获取视频的帧速率和总帧数
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))


    current_frame = 0
    # 设置当前帧的索引
    for s, e in tqdm(start_and_end):
        # 计算裁剪的起始帧和结束帧
        start_frame = int(s * fps)
        end_frame = int(e * fps)
        # 设置视频编解码器并创建输出视频对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(f'clip/{s}-{e}.mp4', fourcc, fps, (int(video.get(3)), int(video.get(4))))


        # 读取并写入视频帧，直到达到结束帧
        while current_frame < end_frame:
            ret, frame = video.read()
            if not ret:
                break
            if current_frame >= start_frame:
                output.write(frame)
            current_frame += 1

        # 释放视频对象和输出对象
    video.release()
    output.release()

    print("视频裁剪完成！")

# 使用示例
input_file = r"C:\Users\ABin\Desktop\S01E03 热带宁静Tropical Serenity_超清 4K.mp4"
start_and_end = [
    [24, 71],
    [76, 178],
    [182, 265],
    [268, 315],
    [322, 366],
    [372, 467],
    [473, 535],
    [550, 610],
    [618, 684],
    [689, 792],
    [800, 880],
    [884, 945],
    [951, 1002],
    [1009, 1116],
    [1127, 1196],
    [1205, 1301],
    [1306, 1381],
    [1386, 1421],
    [1430, 1505],
    [1571, 1628],
    [1636, 1724],
    [1731, 1784],
]
crop_video(input_file, start_and_end)