# trainable-neural-style
Neural Network with Artistic Style Synthesis



### 日志

#### 2016-11-26
Put both generated and training images in same batch through VGG net for efficiency
```
    net, _ = vgg.net(FLAGS.VGG_PATH, tf.concat(0, [generated, images]) - reader.mean_pixel)
```
