# -*- coding: utf-8 -*-

"""
@date: 2023/9/29 下午2:37
@file: t_draw.py
@author: zj
@description:

https://zhuanlan.zhihu.com/p/65330411?utm_id=0
https://blog.csdn.net/qq_15054345/article/details/114685258
https://www.fonts.net.cn/font-36616398905.html
"""

from PIL import Image, ImageDraw, ImageFont

chars = "你好啊 zll"
img_path = "bigwhite.jpg"
# ttf_path = "fonts/HanYiZhongJianHei-2.ttf"
ttf_path = "../assets/fonts/HanYiZhongJianHei-2.ttf"
chars_x, chars_y = 50, 80

# 1. 加载图像文件
# image = Image.open(img_path)
image = Image.open("../assets/1.jpg")
# 2. 加载字体并指定字体大小
# ttf = ImageFont.load_default()  # 默认字体
ttf = ImageFont.truetype(ttf_path, 30)
# 3. 创建绘图对象
img_draw = ImageDraw.Draw(image)
# 4. 在图片上写字
# 第一个参数：指定文字区域的左上角在图片上的位置(x,y)
# 第二个参数：文字内容
# 第三个参数：字体
# 第四个参数：颜色RGB值
img_draw.text((chars_x, chars_y), chars, font=ttf, fill=(255,0,0))
image.show()
# image.save("1.jpg")
