NPZ files (x_train/x_test)
        |
        v
task_incremental_loader._get_loader
  - if numpy -> IQDataGenerator(x, y)
        |
        v
IQDataGenerator.__getitem__
  - if sample is complex or interleaved -> outputs (2, L)
  - if sample already multi-ADC -> preserves shape
        |
        v
Model forward (ResNet1D)
  _prepare_input normalizes:
  - (B, 2L)    -> (B, 2, L)
  - (B, 3, 2L) -> (B, 3, 2, L)
  - (B, 3, 2, L) passes through
  - ambiguous flat -> error
        |
        v
_ResNet1D.forward
  - if 4D (B,3,2,L): AdcIqAdapter -> (B,2,L)
  - if 3D 2-ch: skip adapter
  - if 1-ch: skip adapter
        |
        v
conv1_2 or conv1_1 -> backbone -> logits

Stage	Shape	Where
NPZ / loader	(N, 3, 1024)	Raw samples: 3 ADCs × 1024 time steps
Batch to model	(B, 3, 1024)	e.g. B=256
_prepare_input	(B, 3, 2, 512)	Reshape: treat each ADC’s 1024 as I/Q interleaved → 2×512
AdcIqAdapter (4D path)	(B, 2, 512)	Permute to (B, 2, 3, 512), then einsum with (2,3) weight + bias
Backbone	(B, 2, 512)	conv1 etc. expect 2 channels


uclresm:
        Input shape: (256, 2, 1024)  # dropped ADC1
        Prepared input shape: (256, 2, 1024) 

        Input shape: (256, 3, 2, 512)
        Prepared input shape: (256, 3, 2, 512)
        Using input adapter for 4D input. torch.Size([256, 3, 2, 512])
        Adapter output shape: torch.Size([256, 2, 512])

        eval_tasks():
        Original test shape: (74223, 3, 1024)
        Converted IQ data to 2-channel format, new shape: torch.Size([74223, 3, 2, 512])

radchar_nist:
        Input shape: (256, 2, 512)
        Prepared input shape: (256, 2, 512)


Final Results:- 
 Total Accuracy: 0.3340750793274549 
 Individual Accuracy: [0.24810502288844533, 0.3401808138905009, 0.4139394012034183]
Final Detection Results:- 
 Total Detection: 0.9200129837023292 
 Individual Detection: [0.8874065982734448, 0.9209002323274827, 0.9517321205060599]
Final Detection False Alarm:- 
 Total False Alarm: 0.06133907385188007 
 Individual False Alarm: [0.13328486436637157, 0.0410478199123541, 0.009684537276914523]


 Final Results:- 
 Total Accuracy: 0.4002312054361885 
 Individual Accuracy: [0.21973148672762527, 0.45185212695947125, 0.529110002621469]
Final Detection Results:- 
 Total Detection: 0.8260646878371826 
 Individual Detection: [0.6349624444906684, 0.8846905955168166, 0.9585410235040628]
Final Detection False Alarm:- 
 Total False Alarm: 0.3083816339753715 
 Individual False Alarm: [0.6691370427567497, 0.20874295355594724, 0.047264905613417625]