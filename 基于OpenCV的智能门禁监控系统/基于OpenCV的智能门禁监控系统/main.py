# config# 实验环境：python 3.11 + face_recognition + opencv-python

import cv2
import numpy as np
import os
import shutil
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from datetime import datetime
import csv
import face_recognition
import pickle

# 首先读取config文件，第一行代表当前已经储存的人名个数，接下来每一行是（id，name）标签和对应的人名
id_dict = {}  # 字典里存的是id——name键值对
Total_face_num = 0  # 已经被识别有用户名的人脸个数
known_encodings = {}  # 存储每个人的128维特征向量 {id: encoding}


def init():  # 将config文件内的信息读入到字典中
    f = open('config.txt')
    global Total_face_num
    Total_face_num = int(f.readline())

    for i in range(int(Total_face_num)):
        line = f.readline()
        id_name = line.split(' ')
        id_dict[int(id_name[0])] = id_name[1].strip()
    f.close()

    # 加载已保存的人脸特征向量
    load_encodings()


def load_encodings():
    """从encodings.pkl加载所有已知人脸的特征向量"""
    global known_encodings
    if os.path.exists('encodings.pkl'):
        with open('encodings.pkl', 'rb') as f:
            known_encodings = pickle.load(f)


def save_encodings():
    """保存所有已知人脸的特征向量到encodings.pkl"""
    with open('encodings.pkl', 'wb') as f:
        pickle.dump(known_encodings, f)


init()

# 加载OpenCV人脸检测分类器Haar
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# 打开标号为0的摄像头
camera = cv2.VideoCapture(0)  # 摄像头
success = True
_, img = camera.read()  # 初始化首帧

system_state_lock = 0  # 标志系统状态的量 0表示无子线程在运行 1表示正在刷脸 2表示正在录入新面孔。
# 相当于mutex锁，用于线程同步
new_face_name = ""  # 存储用户自定义的人脸名字

'''
============================================================================================
以上是初始化
============================================================================================
'''


def Get_new_face():
    print("正在从摄像头录入新人脸信息 \n")

    # 存在目录data就清空，不存在就创建，确保最后存在空的data目录
    filepath = "data"
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)

    sample_num = 0  # 已经获得的样本数
    encodings_list = []  # 存储采集到的人脸特征向量

    while True:  # 从视频流读取图片（复用主线程的img）

        global img  # 读取主线程video_loop中更新的img
        frame = img.copy() if img is not None else None
        if frame is None:
            continue

        # 转为灰度图片（Haar级联需要灰度图）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 用OpenCV Haar级联检测人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        # 框选人脸并提取特征
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # 用face_recognition提取特征向量
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encoding = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])
            if len(encoding) > 0:
                encodings_list.append(encoding[0])
                T = Total_face_num
                cv2.imwrite("./data/User." + str(T) + '.' + str(sample_num) + '.jpg', gray[y:y + h, x:x + w])
            sample_num += 1

        pictur_num = 30  # face_recognition只需要少量样本，30张足够

        if sample_num > pictur_num:
            break
        else:  # 控制台内输出进度条
            l = int(sample_num / pictur_num * 50)
            r = int((pictur_num - sample_num) / pictur_num * 50)
            print("\r" + "%{:.1f}".format(sample_num / pictur_num * 100) + "=" * l + "->" + "_" * r, end="")
            var.set("%{:.1f}".format(sample_num / pictur_num * 100))  # 控件可视化进度信息

    # 计算平均特征向量作为该人的模板
    if len(encodings_list) > 0:
        avg_encoding = np.mean(encodings_list, axis=0)
        known_encodings[Total_face_num] = avg_encoding
        save_encodings()
        print(f"\n已保存 {len(encodings_list)} 个样本的特征向量")


def Train_new_face():
    """face_recognition不需要传统训练，特征向量已在Get_new_face中保存"""
    print("\n特征向量已保存，无需额外训练")


def write_config():
    print("新人脸录入完成")
    T = Total_face_num
    global new_face_name
    id_dict[T] = new_face_name

    # 先读取现有内容
    f = open('config.txt', 'r')
    flist = f.readlines()
    f.close()

    # 更新第一行（总人数）
    flist[0] = str(int(flist[0].strip()) + 1) + "\n"

    # 添加新的人脸记录
    flist.append(str(T) + " " + new_face_name + "\n")

    # 写回文件
    f = open('config.txt', 'w')
    f.writelines(flist)
    f.close()


'''
============================================================================================
以上是录入新人脸信息功能的实现
============================================================================================
'''


def scan_face():
    """使用face_recognition进行实时人脸识别"""
    best_match_id = 0
    best_match_conf = 1.0  # face_recognition的距离，越小越匹配
    match_count = {}  # 统计每个id的匹配次数

    # 如果没有已知人脸，直接返回
    if len(known_encodings) == 0:
        print("没有已录入的人脸数据")
        return 0

    frame_count = 0
    max_frames = 30  # 最多检测30帧

    while frame_count < max_frames:
        frame_count += 1

        global system_state_lock
        while system_state_lock == 2:  # 如果正在录入新面孔就阻塞
            print("\r刷脸被录入面容阻塞", end="")
            pass

        global img
        frame = img.copy() if img is not None else None
        if frame is None:
            continue

        # 转为灰度图（Haar检测用）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 用OpenCV Haar级联检测人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for (x, y, w, h) in faces:
            # 用face_recognition提取特征向量
            face_encodings = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])
            if len(face_encodings) == 0:
                continue
            face_encoding = face_encodings[0]

            # 与所有已知人脸比较
            best_id = 0
            best_dist = 1.0

            for pid, pencoding in known_encodings.items():
                # 计算欧氏距离
                dist = face_recognition.face_distance([pencoding], face_encoding)[0]
                if dist < best_dist:
                    best_dist = dist
                    best_id = pid

            # 判断是否匹配（距离<0.6为同一人）
            if best_dist < 0.6:
                user_name = id_dict.get(best_id, "Unknown")
                confidence = round((1 - best_dist) * 100, 1)  # 转换为百分比
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绿色边框
                match_count[best_id] = match_count.get(best_id, 0) + 1
            else:
                user_name = "unknown"
                confidence = 0
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色边框

            # 显示信息
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(user_name), (x + 5, y - 5), font, 1, (0, 0, 255), 1)
            cv2.putText(frame, str(confidence) + "%", (x + 5, y + h - 5), font, 1, (0, 0, 0), 1)

    # 统计结果，返回匹配次数最多的id
    if match_count:
        best_match_id = max(match_count, key=match_count.get)
        print(f"识别结果: id={best_match_id}, 匹配{match_count[best_match_id]}次")
        return best_match_id

    return 0  # 无法识别


'''
============================================================================================
以上是关于刷脸功能的设计
============================================================================================
'''


def save_log(name, status):
    """保存通行日志到CSV文件"""
    log_file = "access_log.csv"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.isfile(log_file)

    with open(log_file, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["时间", "姓名", "状态"])
        writer.writerow([now, name, status])


def show_log_window():
    """显示通行日志查询窗口"""
    log_window = tk.Toplevel(window)
    log_window.title("通行日志管理")
    log_window.geometry("700x500")
    log_window.configure(bg='#f0f4f8')

    # 搜索框
    search_frame = tk.Frame(log_window, bg='#f0f4f8')
    search_frame.pack(fill='x', padx=10, pady=10)

    tk.Label(search_frame, text="搜索姓名:", bg='#f0f4f8', font=('Microsoft YaHei', 10)).pack(side='left')
    search_var = tk.StringVar()
    search_entry = tk.Entry(search_frame, textvariable=search_var, font=('Arial', 10), width=20)
    search_entry.pack(side='left', padx=5)

    # 读取并显示日志
    columns = ("时间", "姓名", "状态")
    tree = ttk.Treeview(log_window, columns=columns, show='headings', height=20)

    tree.heading("时间", text="时间")
    tree.heading("姓名", text="姓名")
    tree.heading("状态", text="状态")

    tree.column("时间", width=200, anchor='center')
    tree.column("姓名", width=150, anchor='center')
    tree.column("状态", width=100, anchor='center')

    # 滚动条
    scrollbar = ttk.Scrollbar(log_window, orient='vertical', command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)

    def load_logs(keyword=""):
        """加载日志数据"""
        for item in tree.get_children():
            tree.delete(item)

        log_file = "access_log.csv"
        if not os.path.exists(log_file):
            return

        with open(log_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header = next(reader)  # 跳过表头
            for row in reader:
                if len(row) >= 3:
                    name = row[1].strip()
                    if keyword == "" or keyword.lower() in name.lower():
                        tree.insert('', 'end', values=(row[0].strip(), name, row[2].strip()))

    def search_logs():
        load_logs(search_var.get().strip())

    def clear_logs():
        """清空日志文件"""
        if messagebox.askyesno("确认", "确定要清空所有通行日志吗？"):
            log_file = "access_log.csv"
            if os.path.exists(log_file):
                os.remove(log_file)
            load_logs()
            messagebox.showinfo("提示", "日志已清空")

    tk.Button(search_frame, text="搜索", command=search_logs, bg='#2196F3', fg='white',
              font=('Microsoft YaHei', 9), relief='flat').pack(side='left', padx=5)
    tk.Button(search_frame, text="刷新", command=lambda: load_logs(), bg='#4CAF50', fg='white',
              font=('Microsoft YaHei', 9), relief='flat').pack(side='left', padx=5)
    tk.Button(search_frame, text="清空日志", command=clear_logs, bg='#f44336', fg='white',
              font=('Microsoft YaHei', 9), relief='flat').pack(side='right', padx=5)

    tree.pack(side='left', fill='both', expand=True, padx=10, pady=5)
    scrollbar.pack(side='right', fill='y', pady=5)

    load_logs()


def f_scan_face_thread():
    var.set('刷脸中...')
    ans = scan_face()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if ans == 0:
        result = f"无法识别  [{now}]"
        print(result)
        var.set(result)
        save_log("未知人员", "拒绝通行")
    else:
        result = f"{id_dict[ans]}  [{now}]"
        print(result)
        var.set(result)
        save_log(id_dict[ans], "允许通行")

    global system_state_lock
    print("锁被释放0")
    system_state_lock = 0  # 修改system_state_lock,释放资源


def f_scan_face():
    global system_state_lock
    print("\n当前锁的值为：" + str(system_state_lock))
    if system_state_lock == 1:
        print("阻塞，因为正在刷脸")
        return 0
    elif system_state_lock == 2:  # 如果正在录入新面孔就阻塞
        print("\n刷脸被录入面容阻塞\n")
        return 0
    system_state_lock = 1
    p = threading.Thread(target=f_scan_face_thread)
    p.daemon = True
    p.start()


def f_rec_face_thread():
    var.set('录入')
    cv2.destroyAllWindows()
    global Total_face_num
    global new_face_name
    Total_face_num += 1
    Get_new_face()  # 采集新人脸
    print("采集完毕，开始保存特征向量")
    global system_state_lock  # 采集完就可以解开锁
    print("锁被释放0")
    system_state_lock = 0

    Train_new_face()  # face_recognition无需训练
    write_config()  # 修改配置文件


def f_rec_face():
    global system_state_lock
    global new_face_name
    print("当前锁的值为：" + str(system_state_lock))
    if system_state_lock == 2:
        print("阻塞，因为正在录入面容")
        return 0

    # 弹出输入框让用户输入名字
    name_window = tk.Toplevel(window)
    name_window.title("录入人脸 - 输入姓名")
    name_window.geometry("300x150")

    tk.Label(name_window, text="请输入人脸姓名:", font=('Arial', 12)).pack(pady=20)
    name_entry = tk.Entry(name_window, font=('Arial', 12), width=20)
    name_entry.pack()

    def confirm_name():
        global new_face_name
        new_face_name = name_entry.get().strip()
        if not new_face_name:
            new_face_name = "未知用户" + str(Total_face_num + 1)
        name_window.destroy()

        # 开始录入流程
        global system_state_lock
        system_state_lock = 2
        print("改为2", end="")
        print("当前锁的值为：" + str(system_state_lock))

        p = threading.Thread(target=f_rec_face_thread)
        p.daemon = True
        p.start()

    tk.Button(name_window, text="确认", font=('Arial', 10), command=confirm_name).pack(pady=10)
    name_entry.focus()


def f_exit():  # 退出按钮
    exit()


'''
============================================================================================
以上是关于多线程的设计
============================================================================================
'''

window = tk.Tk()
window.title('人脸识别系统')  # 窗口标题
window.geometry('800x550')  # 这里的乘是小x
window.configure(bg='#FDF0F5')
# 在图形界面上设定标签，类似于一个提示窗口的作用
var = tk.StringVar()
l = tk.Label(window, textvariable=var, bg='#BAAFD1', fg='white', font=('Arial', 12), width=40, height=4)
l.pack()  # 放置l控件
var.set('⭐人脸识别系统⭐')

# 在窗口界面设置放置Button按键并绑定处理函数
button_a = tk.Button(window, text='开始刷脸', font=('Arial', 12), width=10, height=2, command=f_scan_face)
button_a.place(x=620, y=110)

button_b = tk.Button(window, text='录入人脸', font=('Arial', 12), width=10, height=2, command=f_rec_face)
button_b.place(x=620, y=210)

button_d = tk.Button(window, text='通行日志', font=('Arial', 12), width=10, height=2, command=show_log_window)
button_d.place(x=620, y=310)

button_c = tk.Button(window, text='退出', font=('Arial', 12), width=10, height=2, command=f_exit)
button_c.place(x=620, y=410)

panel = tk.Label(window, width=500, height=350)  # 摄像头模块大小
panel.place(x=50, y=100)  # 摄像头模块的位置
window.config(cursor="arrow")


def video_loop():  # 用于在label内动态展示摄像头内容（摄像头嵌入控件）
    global success
    global img
    global system_state_lock
    # 始终读取摄像头，确保工作线程能获取最新帧
    success, img = camera.read()
    cv2.waitKey(1)

    if system_state_lock == 0 and success and img is not None:
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        current_image = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=current_image)
        panel.imgtk = imgtk
        panel.config(image=imgtk)
    window.after(1, video_loop)


video_loop()

#  窗口循环，用于显示
window.mainloop()

'''
============================================================================================
以上是关于界面的设计
============================================================================================
'''
