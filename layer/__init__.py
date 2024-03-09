from .feedforward import ConvNet1d, SensoryEncoder, MLP
from .MLP_230412 import MLP_230412
from .MLP_230412_noFB import MLP_230412_noFB
from .MLP_230726_noFB import MLP_230726_noFB
from .MLP_230817_atOnset import MLP_230817_atOnset

# from .recurrent import StackedGRU, StackedLayerNormGRU
# from .recurrent_230412 import StackedGRU_230412, StackedLayerNormGRU
from .recurrent_231027 import StackedGRU_231027, StackedLayerNormGRU
from .recurrent_231027_forInfer import StackedGRU_231027_forInfer #, StackedLayerNormGRU
from .recurrent_231027_forInfer_300ms import StackedGRU_231027_forInfer_300ms #, StackedLayerNormGRU

from .spine import SpinalCordCircuit
from .spine_230119 import SpinalCordCircuit_230119
from .spine_230202 import SpinalCordCircuit_230202
from .spine_230202_noWclip import SpinalCordCircuit_230202_noWclip
from .spine_230202_weightShare import SpinalCordCircuit_230202_weightShare
from .spine_230202_positiveW import SpinalCordCircuit_230202_positiveW
from .spine_230209 import SpinalCordCircuit_230209
from .spine_230216 import SpinalCordCircuit_230216
from .spine_230223 import SpinalCordCircuit_230223
from .spine_230227 import SpinalCordCircuit_230227
from .spine_230307 import SpinalCordCircuit_230307
from .spine_230309 import SpinalCordCircuit_230309
from .spine_230310 import SpinalCordCircuit_230310
from .spine_230313 import SpinalCordCircuit_230313
from .spine_230313_withFB import SpinalCordCircuit_230313_withFB
from .spine_230313_withKineFB import SpinalCordCircuit_230313_withKineFB
from .spine_230314_withKineFB import SpinalCordCircuit_230314_withKineFB
from .spine_230531_withKineFB import SpinalCordCircuit_230531_withKineFB
from .spine_230531_withKineFB_woBclip import SpinalCordCircuit_230531_withKineFB_woBclip
from .spine_230531_withKineFB_1act import SpinalCordCircuit_230531_withKineFB_1act
from .spine_230605_simpleAct import SpinalCordCircuit_230605_simpleAct
from .spine_230606_simpleActReal import SpinalCordCircuit_230606_simpleActReal
from .spine_230605_simpleAct_synthetic import SpinalCordCircuit_230605_simpleAct_synthetic
from .spine_230627_unified import SpinalCordCircuit_230627_unified
from .spine_230630_unified import SpinalCordCircuit_230630_unified
from .spine_230703_unified import SpinalCordCircuit_230703_unified
from .spine_230703_unified_t25 import SpinalCordCircuit_230703_unified_t25
from .spine_230703_unified_t25_1 import SpinalCordCircuit_230703_unified_t25_1
from .spine_230703_unified_t25_2 import SpinalCordCircuit_230703_unified_t25_2
from .spine_230703_unified_t25_3 import SpinalCordCircuit_230703_unified_t25_3
from .spine_230703_unified_t25_4 import SpinalCordCircuit_230703_unified_t25_4
from .spine_230703_unified_t25_5 import SpinalCordCircuit_230703_unified_t25_5
from .spine_230718_unified_t25_5 import SpinalCordCircuit_230718_unified_t25_5
from .spine_230718_unified_t25_5_old import SpinalCordCircuit_230718_unified_t25_5_old
from .spine_230719_unified_t25_5 import SpinalCordCircuit_230719_unified_t25_5
from .spine_230720_unified_t25_5 import SpinalCordCircuit_230720_unified_t25_5
from .spine_230720_unified_t25_5_mnsFB import SpinalCordCircuit_230720_unified_t25_5_mnsFB
from .spine_230720_unified_t25_5_Kine import SpinalCordCircuit_230720_unified_t25_5_Kine
from .spine_230727_unified_t25_5_mnsFB import SpinalCordCircuit_230727_unified_t25_5_mnsFB
from .spine_230802_unified_t25_5_mnsFB import SpinalCordCircuit_230802_unified_t25_5_mnsFB
from .spine_230807_unified_t25_5_mnsFB import SpinalCordCircuit_230807_unified_t25_5_mnsFB
from .spine_230817_unified_t25_5_mnsFB import SpinalCordCircuit_230817_unified_t25_5_mnsFB
from .spine_230817_unified_t25_5_mnsFB_atOnset import SpinalCordCircuit_230817_unified_t25_5_mnsFB_atOnset
from .spine_230817_unified_t25_5_mnsFB_preOnset import SpinalCordCircuit_230817_unified_t25_5_mnsFB_preOnset
from .spine_230817_unified_t25_5_mnsFB_gradual import SpinalCordCircuit_230817_unified_t25_5_mnsFB_gradual
from .spine_230926_unified_t25_5_mnsFB_atOnset import SpinalCordCircuit_230926_unified_t25_5_mnsFB_atOnset
from .spine_230926_unified_t25_5_mnsFB_atOnset_forInfer import SpinalCordCircuit_230926_unified_t25_5_mnsFB_atOnset_forInfer
from .spine_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010 import SpinalCordCircuit_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010
from .spine_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010_decinc import SpinalCordCircuit_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010_decinc
from .spine_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010_decinc_area import SpinalCordCircuit_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010_decinc_area
from .spine_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010_decinc_300ms import SpinalCordCircuit_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010_decinc_300ms
from .spine_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010_decinc_xy import SpinalCordCircuit_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010_decinc_xy
from .spine_230926_unified_t25_5_mnsFB_atOnset_forIllu import SpinalCordCircuit_230926_unified_t25_5_mnsFB_atOnset_forIllu


def get_layer(name):
    return {
        "ConvNet1d": ConvNet1d,
        'SensoryEncoder': SensoryEncoder,
#         "GRU_230412": StackedGRU_230412,
        "GRU_231027": StackedGRU_231027,
        "GRU_231027_forInfer": StackedGRU_231027_forInfer,
        "GRU_231027_forInfer_300ms": StackedGRU_231027_forInfer_300ms,
        "LayerNormGRU": StackedLayerNormGRU,
        "SCC": SpinalCordCircuit,
        "SCC_230119": SpinalCordCircuit_230119,
        "SCC_230202": SpinalCordCircuit_230202,
        "SCC_230202_noWclip": SpinalCordCircuit_230202_noWclip,
        "SCC_230202_weightShare": SpinalCordCircuit_230202_weightShare,
        "SCC_230202_positiveW": SpinalCordCircuit_230202_positiveW,
        "MLP": MLP,
        "MLP_230412": MLP_230412,
        "MLP_230412_noFB": MLP_230412_noFB,
        "MLP_230726_noFB": MLP_230726_noFB,
        "MLP_230817_atOnset": MLP_230817_atOnset,
        "SCC_230209": SpinalCordCircuit_230209,
        "SCC_230216": SpinalCordCircuit_230216,
        "SCC_230223": SpinalCordCircuit_230223,
        "SCC_230227": SpinalCordCircuit_230227,
        "SCC_230307": SpinalCordCircuit_230307,
        "SCC_230309": SpinalCordCircuit_230309,
        "SCC_230310": SpinalCordCircuit_230310,
        "SCC_230313": SpinalCordCircuit_230313,
        "SCC_230313_withFB": SpinalCordCircuit_230313_withFB,
        "SCC_230313_withKineFB": SpinalCordCircuit_230313_withKineFB,
        "SCC_230314_withKineFB": SpinalCordCircuit_230314_withKineFB,
        "SCC_230531_withKineFB": SpinalCordCircuit_230531_withKineFB,
        "SCC_230531_withKineFB_woBclip": SpinalCordCircuit_230531_withKineFB_woBclip,
        "SCC_230531_withKineFB_1act": SpinalCordCircuit_230531_withKineFB_1act,
        "SCC_230605_simpleAct": SpinalCordCircuit_230605_simpleAct,
        "SCC_230606_simpleActReal": SpinalCordCircuit_230606_simpleActReal,
        "SCC_230605_simpleAct_synthetic": SpinalCordCircuit_230605_simpleAct_synthetic,
        "SCC_230627_unified": SpinalCordCircuit_230627_unified,
        "SCC_230630_unified": SpinalCordCircuit_230630_unified,
        "SCC_230703_unified": SpinalCordCircuit_230703_unified,
        "SCC_230703_unified_t25": SpinalCordCircuit_230703_unified_t25,
        "SCC_230703_unified_t25_1": SpinalCordCircuit_230703_unified_t25_1,
        "SCC_230703_unified_t25_2": SpinalCordCircuit_230703_unified_t25_2,
        "SCC_230703_unified_t25_3": SpinalCordCircuit_230703_unified_t25_3,
        "SCC_230703_unified_t25_4": SpinalCordCircuit_230703_unified_t25_4,
        "SCC_230703_unified_t25_5": SpinalCordCircuit_230703_unified_t25_5,
        "SCC_230718_unified_t25_5": SpinalCordCircuit_230718_unified_t25_5,
        "SCC_230718_unified_t25_5_old": SpinalCordCircuit_230718_unified_t25_5_old,
        "SCC_230719_unified_t25_5": SpinalCordCircuit_230719_unified_t25_5,
        "SCC_230720_unified_t25_5": SpinalCordCircuit_230720_unified_t25_5,
        "SCC_230720_unified_t25_5_mnsFB": SpinalCordCircuit_230720_unified_t25_5_mnsFB,
        "SCC_230720_unified_t25_5_Kine": SpinalCordCircuit_230720_unified_t25_5_Kine,
        "SCC_230727_unified_t25_5_mnsFB": SpinalCordCircuit_230727_unified_t25_5_mnsFB,
        "SCC_230802_unified_t25_5_mnsFB": SpinalCordCircuit_230802_unified_t25_5_mnsFB,
        "SCC_230807_unified_t25_5_mnsFB": SpinalCordCircuit_230807_unified_t25_5_mnsFB,
        "SCC_230817_unified_t25_5_mnsFB": SpinalCordCircuit_230817_unified_t25_5_mnsFB,
        "SCC_230817_unified_t25_5_mnsFB_atOnset": SpinalCordCircuit_230817_unified_t25_5_mnsFB_atOnset,
        "SCC_230817_unified_t25_5_mnsFB_preOnset": SpinalCordCircuit_230817_unified_t25_5_mnsFB_preOnset,  
        "SCC_230817_unified_t25_5_mnsFB_gradual": SpinalCordCircuit_230817_unified_t25_5_mnsFB_gradual,
        "SCC_230926_unified_t25_5_mnsFB_atOnset": SpinalCordCircuit_230926_unified_t25_5_mnsFB_atOnset,
        "SCC_230926_unified_t25_5_mnsFB_atOnset_forInfer": SpinalCordCircuit_230926_unified_t25_5_mnsFB_atOnset_forInfer,
        "SCC_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010": SpinalCordCircuit_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010,
        "SCC_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010_decinc": SpinalCordCircuit_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010_decinc,
        "SCC_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010_decinc_area": SpinalCordCircuit_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010_decinc_area,
        "SCC_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010_decinc_300ms": SpinalCordCircuit_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010_decinc_300ms,
        "SCC_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010_decinc_xy": SpinalCordCircuit_230926_unified_t25_5_mnsFB_atOnset_forInfer_v231010_decinc_xy,
        "SCC_230926_unified_t25_5_mnsFB_atOnset_forIllu": SpinalCordCircuit_230926_unified_t25_5_mnsFB_atOnset_forIllu,
    }[name]