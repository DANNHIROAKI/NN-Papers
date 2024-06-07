---
name: CRF-RNN Semantic Image Segmentation Model trained on COCO-VOC
caffemodel: TVG_CRFRNN_COCO_VOC.caffemodel
caffemodel_url: http://goo.gl/j7PrPZ
license: Non-commercial, for commercial use, please contact crfasrnn@gmail.com
sha1: bfda5c5149d566aa56695789fa9a08e7a7f3070a
---

This model is for the ICCV paper titled "Conditional Random Fields as Recurrent Neural Networks".

With this model, you should get 73.0 mean-IOU score on a reduced validation set(346 Images http://www.robots.ox.ac.uk/~szheng/Res_CRFRNN/seg12valid.txt) on PASCAL VOC 2012. Notice that without this CRF-RNN layer, our end-to-end-trained FCN-8s model gives 69.85 mean-IOU score (mean pix. accuracy:92.94, pixel accuracy: 78.80) on the this reduced validation set of PASCAL VOC 2012.

The input is expected in BGR color channel order, with the following per-channel mean substracted:
B: 104.00698793 G: 116.66876762 R: 122.67891434

Demo website is <http://crfasrnn.torr.vision>.

This model was trained by 
Shuai Zheng @bittnt
Sadeep Jayasumana @sadeepj
Bernardino Romera-Paredes @bernard24

Supervisor:
Philip Torr : <philip.torr@eng.ox.ac.uk>

## License
This model is for the non-commercial. For other use, please contact <crfasrnn@gmail.com>

