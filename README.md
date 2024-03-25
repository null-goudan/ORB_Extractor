## ORB特征提取器
通过对OpenCV中的ORB特征点提取类进行修改，对图像进行分块提取，而后划分节点，使得每个节点中保存的特征点性能是该节点所有特征点中最好的。（取自ORB-SLAM算法中ORB提取器）

### 效果 （GridOrbKpsimg窗口为本算法、OrbKpsimg为Opencv本体算法）
![image](https://github.com/null-goudan/ORB_Extractor/assets/74131166/4f32b224-544c-4853-a169-a2f3dcd05e98)
![image](https://github.com/null-goudan/ORB_Extractor/assets/74131166/7fc49487-e38d-41df-8799-a7a1a186d652)

**可以看到原方法提取到的特征点聚集在某个区域，甚至有些边角信息没有提取到（如地面上的缝隙、桌边、桌上的比较微弱纹理这样的信息）**

### 为什么需要这个算法

**主要原因**:为了SLAM的跟踪更加鲁棒,特征点需要更加的均匀、精准、微弱纹理提取。这样对相机位姿追踪有鲁棒性的提升

类比小故事（算法为什么Work）:
将铺满苹果的桌子进行画格子，然后每个格子中就会有不同数量的苹果，在每个格子中选出最好吃的苹果，格子中其他的苹果全部扔掉。（虽然有点可惜，但是大局为重嘛），那么原先摆满苹果的桌子，现在就剩下每个格子一个苹果的桌子，尽管苹果少了很多，但是剩下的都是精英，极品

![image](https://github.com/null-goudan/ORB_Extractor/assets/74131166/d7e20eef-d199-49a8-b315-f614149cbaed)
![image](https://github.com/null-goudan/ORB_Extractor/assets/74131166/568d0891-8622-4ae8-95cd-4efebc14fdb7)

原生态的ORB特征提取的方法，他主要是通过阈值条件选出所有满足条件的ORB描述子，然后计算所有描述子的响应强度并排序M，根据输入要求的特征点数量N，取M中前N个描述子，即响应值最大的前N个描述子。显然，这种提取的方法会导致特征点的分布非常不均匀。而这也会影响到SLAM系统中定位的精度

### 算法流程

　　1. 输入图像，并对输入图像进行预处理，将其转换成灰度图像；

　　2. 初始化参数，包括特征点数量nfeatures，尺度scaleFactor，金字塔层数nlevel，初始阈值iniThFAST，最小阈值minThFAST等参数；

　　3. 计算金字塔图像，源码中使用8层金字塔，尺度因子为1.2，则通过对原图像进行不同层次的resize，可以获得8层金字塔的图像；

　　4. 计算特征点：

　　　　1）将图像分割成网格，每个网格大小为W*W=30*30像素；

　　　　2）遍历每个网格；

　　　　3）对每个网格提取FAST关键点，先用初始阈值iniThFAST提取，若提取不到关键点，则改用最小阈值minThFAST提取。（注意，初始阈值一般比最小阈值大）

　　5. 对所有提取到的关键点利用八叉树的形式进行划分：

　　　　1）按照像素宽和像素高的比值作为初始的节点数量，并将关键点坐标落在对应节点内的关键点分配入节点中；

　　　　2）根据每个节点中存在的特征点数量作为判断依据，如果当前节点只有1个关键点，则停止分割。否则继续等分成4份；

　　　　3）按照上述方法不断划分下去，如图3所示，可见出现一个八叉树的结构，终止条件是节点的数目Lnode大于等于要求的特征点数量nfeatures；

　　　　4）对满足条件的节点进行遍历，在每个节点中保存响应值最大的关键点，保证特征点的高性能；

　　6. 对上述所保存的所有节点中的特征点计算主方向，利用灰度质心的方法计算主方向，上一讲中我们已经讲解过方法，这讲就不再赘述了；

　　7. 对图像中每个关键点计算其描述子，值得注意的是，为了将主方向融入BRIEF中，在计算描述子时，ORB将pattern进行旋转，使得其具备旋转不变性

### 如何使用本源码
第一种办法： 当时测试用

```c
./test.sh
```
第二种办法:
```c
cd Build
cmake ..
make
cd ../bin
./orb_extractor ./1.png
```
  
