# DCT0文件夹代码使用说明
## 文件内容：
#### 本文件包含4部分内容，分别是DCT相位恢复，迭代DCT相位恢复，测试图像生成与相位恢复主函数四部分
### 不足：
#### 由于没有考虑实际情况下相位恢复时采样图像自带的噪点问题，相关算法在实际情况下的应用情况堪忧
#### 待给出除了热力图外的图像相位恢复误差统计情况分析
#### 有一些参数的实际情况下取值待完善，如more,threshold等，实际情况下可能的取值与现在不同

##### 题外话：
##### TIE_Functions.py文件包含了一些杂七杂八的次要函数，如评价图像是否符合要求的Estimate部分