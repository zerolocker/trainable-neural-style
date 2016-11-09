# trainable-neural-style
Neural Network with Artistic Style Synthesis



### 日志

#### 2016-11-8
**TODO** Inspect two things: 
  1. the relative magnitude of style_loss and content_loss in stylenet
  2. whether stylenet has the "grey" image problem (i.e. the range of values in generated image falls out of [0,1])


#### 2016-11-5
+ 仅通过看代码发现github repo `stylenet` 里面的stylenet_core.py的 get_style_cost_gram实现有问题。因为函数l2_norm_cost里已经除了size^2了， 而这里又除一遍。这样会造成这个loss十分小。 （更新：但是我发现其他几个repo都是这样子的实现。所以估计是对的）
