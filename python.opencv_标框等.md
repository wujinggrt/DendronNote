---
id: koi0o1958pc5xlv6k70eind
title: Opencv_标框等
desc: ''
updated: 1753113785134
created: 1753113734724
---

```py
import cv2
import numpy as np
import os
import time

class CameraAnnotator:
    def __init__(self, camera_index=0, window_name="Camera Annotator"):
        """
        初始化相机标注工具
        
        参数:
        camera_index - 相机索引 (默认为0)
        window_name - 窗口名称
        """
        self.camera_index = camera_index
        self.window_name = window_name
        self.cap = None
        self.frame = None
        self.annotated_frame = None
        self.drawing = False
        self.bboxes = []  # 存储所有标注框 [(x1, y1, x2, y2)]
        self.current_bbox = None  # 当前正在绘制的框 (x1, y1, x2, y2)
        self.start_point = None
        self.save_count = 0
        
        # 创建保存目录
        self.save_dir = "annotated_images"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 初始化相机
        self.init_camera()
        
        # 设置窗口和鼠标回调
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # 打印操作说明
        print("=" * 60)
        print("相机标注工具操作指南:")
        print("1. 鼠标左键拖拽: 绘制边界框")
        print("2. 按 's': 保存当前帧和标注框")
        print("3. 按 'd': 删除最后一个标注框")
        print("4. 按 'c': 清除所有标注框")
        print("5. 按 'r': 重置相机")
        print("6. 按 'q': 退出程序")
        print("=" * 60)
    
    def init_camera(self):
        """初始化相机"""
        # 尝试打开相机
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print(f"错误: 无法打开相机索引 {self.camera_index}")
            print("请检查相机连接或尝试其他索引")
            self.cap = None
            return False
        
        # 设置相机分辨率 (根据相机支持调整)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print(f"相机 {self.camera_index} 已成功初始化")
        return True
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标事件回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 开始绘制新框
            self.drawing = True
            self.start_point = (x, y)
            self.current_bbox = [x, y, x, y]
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # 更新当前框的大小
                self.current_bbox[2] = x
                self.current_bbox[3] = y
        
        elif event == cv2.EVENT_LBUTTONUP:
            # 完成绘制
            self.drawing = False
            x1, y1, x2, y2 = self.current_bbox
            
            # 确保框有有效大小
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                # 确保左上角和右下角正确
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                self.bboxes.append((x1, y1, x2, y2))
                print(f"添加标注框: ({x1}, {y1}) - ({x2}, {y2})")
            else:
                print("标注框太小，已忽略")
            
            self.current_bbox = None
    
    def process_frame(self):
        """处理当前帧并绘制标注框"""
        if self.frame is None:
            return None
        
        # 创建用于显示的帧副本
        display_frame = self.frame.copy()
        
        # 绘制所有已保存的标注框
        for i, (x1, y1, x2, y2) in enumerate(self.bboxes):
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 在框左上角显示编号
            cv2.putText(display_frame, str(i+1), (x1+5, y1+20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 绘制当前正在绘制的框（临时框）
        if self.drawing and self.current_bbox is not None:
            x1, y1, x2, y2 = self.current_bbox
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # 显示状态信息
        status_text = f"标注框: {len(self.bboxes)} | 按 's' 保存 | 按 'q' 退出"
        cv2.putText(display_frame, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return display_frame
    
    def save_current_frame(self):
        """保存当前帧和标注信息"""
        if self.frame is None:
            print("错误: 没有可用的帧")
            return
        
        self.save_count += 1
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 保存图像
        img_filename = os.path.join(self.save_dir, f"annotated_{timestamp}_{self.save_count}.jpg")
        cv2.imwrite(img_filename, self.frame)
        
        # 保存标注信息
        annot_filename = os.path.join(self.save_dir, f"annotations_{timestamp}_{self.save_count}.txt")
        with open(annot_filename, 'w') as f:
            for i, (x1, y1, x2, y2) in enumerate(self.bboxes):
                f.write(f"{i+1},{x1},{y1},{x2},{y2}\n")
        
        print(f"保存成功! 图像: {img_filename}")
        print(f"        标注: {annot_filename}")
    
    def run(self):
        """主运行循环"""
        if self.cap is None:
            print("无法启动相机，退出程序")
            return
        
        while True:
            # 捕获帧
            ret, self.frame = self.cap.read()
            if not ret:
                print("错误: 无法从相机读取帧")
                time.sleep(0.1)
                continue
            
            # 处理帧并显示
            self.annotated_frame = self.process_frame()
            cv2.imshow(self.window_name, self.annotated_frame)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 退出
                break
            elif key == ord('s'):  # 保存
                self.save_current_frame()
            elif key == ord('d'):  # 删除最后一个框
                if self.bboxes:
                    removed = self.bboxes.pop()
                    print(f"删除标注框: {removed}")
                else:
                    print("没有可删除的标注框")
            elif key == ord('c'):  # 清除所有框
                self.bboxes = []
                print("已清除所有标注框")
            elif key == ord('r'):  # 重置相机
                print("重置相机...")
                self.cap.release()
                time.sleep(1)  # 等待相机释放
                if self.init_camera():
                    print("相机重置成功")
                else:
                    print("相机重置失败，退出程序")
                    break
        
        # 清理
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("程序已退出")

# 使用示例
if __name__ == "__main__":
    # 尝试不同的相机索引 (0, 1, 2等)
    camera_index = 0
    
    # 创建并运行标注工具
    annotator = CameraAnnotator(camera_index)
    annotator.run()
```

使用后，可以用鼠标拖拽表框，d 删除。

## Ref and Tag