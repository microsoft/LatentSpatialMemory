# ruff: noqa
from .iqa_pytorch.qalign_metrics import QAlignAestheticMetric, QAlignQualityMetric
from .iqa_pytorch.clip_iqa_metrics import (
    CLIPImageQualityAssessmentMetric as IQACLIPImageQualityAssessmentMetric,
    CLIPImageQualityAssessmentPlusMetric,
    CLIPImageQualityAssessmentPlusRN50_512Metric,
    CLIPimageQualityAssessmentPlusVITL14_512Metric,
)
from .iqa_pytorch.clip_score_metrics import CLIPScoreMetric as IQACLIPScoreMetric
from .iqa_pytorch.clip_aesthetic_metrics import (
    CLIPAestheticScoreMetric as IQACLIPAestheticScoreMetric,
)
from .iqa_pytorch.musiq_metrics import MultiScaleImageQualityMetric

# third party
from .third_party.clip_mlp_aesthetic_metrics import CLIPMLPAestheticScoreMetric
from .third_party.clip_consistency_metrics import CLIPConsistencyMetric
from .third_party.dino_consistency_metrics import DINOConsistencyMetric

from .third_party.camera_error_metrics import CameraErrorMetric
from .third_party.flow_aepe_metrics import OpticalFlowAverageEndPointErrorMetric
from .third_party.gram_matrix_metrics import GramMatrixMetric
from .third_party.qalign_video_metrics import (
    QAlignVideoAestheticMetric,
    QAlignVideoQualityMetric,
)
from .third_party.object_detection_metrics import ObjectDetectionMetric
from .third_party.reprojection_error_metrics import ReprojectionErrorMetric
from .third_party.flow_metrics import OpticalFlowMetric
from .third_party.motion_accuracy_metrics import MotionAccuracyMetric
from .third_party.motion_smoothness_metrics import MotionSmoothnessMetric

# torchmetrics
from .torchmetrics.clip_iqa_metrics import CLIPImageQualityAssessmentMetric
from .torchmetrics.clip_score_metrics import CLIPScoreMetric
