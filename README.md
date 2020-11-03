# RSNA Bone Age

# file hierarchy

```bash
.
├── input
│   ├── boneage-test-dataset
│   │   └── boneage-test-dataset
│   │       ├── 4360.png
│   │       ├── 4361.png
│   ├── boneage-test-dataset.csv
│   ├── boneage-training-dataset
│   │   └── boneage-training-dataset
│   │       ├── 10000.png
│   │       ├── 10001.png
```

# before running any script
Download the neural network weights inceptionV3:
```bash
wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
```
