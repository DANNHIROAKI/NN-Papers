# evaluation on VS tasks
from .inference_image_generic_seg import InferenceImageGenericSegmentation
from .inference_video_vis import InferenceVideoVIS
from .inference_video_vis_fast import InferenceVideoVISFast
from .inference_video_vps import InferenceVideoVPS
from .inference_video_vos import InferenceVideoVOS
from .inference_video_entity import InferenceVideoEntity

# semantic extraction for raw videos with .mp4
from .inference_video_semantic_extraction import InferenceVideoSemanticExtraction