# RSNA Bone Age

If you are working locally, download the data:
https://www.kaggle.com/kmader/rsna-bone-age?select=boneage-training-dataset.csv

## File hierarchy

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

## Before running any script...

Download the neural network weights inceptionV3:

### Unix:
```bash
wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
```


### Windows Powershell:
```bash
$url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

$outputDir = "$pwd\InceptionV3\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

$wc = New-Object System.Net.WebClient

$wc.DownloadFile($url, $outputDir)
```


If you are part of UVG, download the model weights. Otherwise, train the models again.

https://drive.google.com/drive/folders/1eqAOPh8HmC3icYme-CO_VzGjllQ_vI1e?usp=sharing
