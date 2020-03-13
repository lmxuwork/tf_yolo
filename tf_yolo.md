## 使用Tensorflow实现YoLo模型

### YOLO系列
- 之前 Faster-RCNN、Fast-RCNN等是将整体流程划分为区域提取和目标分类两部分进行的，这样做的特点是精度高，速度慢。
- YOLO(You Only Look Once),是端到端的目标检测算法。
  - YOLOv1
    - 核心思想
      - 将目标检测作为回归问题解决
      - 改革了区域建议框式检测框架: RCNN系列均需要生成建议框，在建议框上进行分类与回归，但建议框之间有重叠，这会带来很多重复工作。YOLO将全图划分为SXS的格子，每个格子负责中心在该格子的目标检测，采用一次性预测所有格子所含目标的bbox、定位置信度以及所有类别概率向量来将问题一次性解决(one-shot)。
    - 过程
      - 1、将图像resize到448*448作为神经网络的输入
      - 2、将 $448*448*3$ 的图像经过多层卷积(24层卷积)和一个全连接层后输出 $4096$ 维的feature map(特征图)。
      - 3、看到最后一个全连接层，这里实现了YOLOv1最为关键的一步,将图像划分为S×S个网格（grid cell）。即将图像划分成 [公式] 个网格，而每个网格预测位置误差和分类误差，即B个bounding box的置信度和box的位置参数(x,y,w,h)和C个条件类别概率P。
        - 如果某个object落入某个grid cell,那么这个grid cell就对该object负责。同时，每个grid cell预测B个类别的bounding box的位置和置信度。这个置信度并不只是该bounding box是待检测目标的概率，而是该bounding box是待检测目标的概率乘上该bounding box和真实位置的IoU的积。通过乘上这个交并比，反映出该bounding box预测位置的精度。
    　　　　　　$confidence = P(object)*IoU_{pred}^{truth} $
      　　　　　　　$S=7，B=2,$　C : 表示多少个类别
        - 每个bounding box对应于5个输出，分别是x,y,w,h和上述提到的置信度。其中，x,y代表bounding box的中心离开其所在grid cell边界的偏移。w,h代表bounding box真实宽高相对于整幅图像的比例。x,y,w,h这几个参数都已经被bounded到了区间[0,1]上。除此以外，每个grid cell还产生C个条件概率，P(Classi|Object)。注意，我们不管B的大小，每个grid cell只产生一组这样的概率。在test的非极大值抑制阶段，对于每个bounding box，我们应该按照下式衡量该框是否应该予以保留。
      - 4、进行反向传播修正整个网络模型的参数。
      - ![]