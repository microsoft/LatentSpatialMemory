from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = ''
_CN.suffix =''
_CN.gamma = 0.8
_CN.max_flow = 400
_CN.batch_size = 8
_CN.sum_freq = 100
_CN.val_freq = 5000
_CN.image_size = [368, 496]
_CN.add_noise = False
_CN.use_smoothl1 = False
_CN.critical_params = []

### change the path here
_CN.restore_ckpt = "logs/PATH-TO-FINAL-FILE/final"

_CN.transformer = 'percostformer3'

_CN.percostformer3 = CN()
_CN.percostformer3.pe = 'linear'
_CN.percostformer3.dropout = 0.0
_CN.percostformer3.droppath = 0.0
_CN.percostformer3.encoder_latent_dim = 256 # in twins, this is 256
_CN.percostformer3.query_latent_dim = 64
_CN.percostformer3.cost_latent_input_dim = 64
_CN.percostformer3.cost_latent_token_num = 8
_CN.percostformer3.cost_latent_dim = 128
_CN.percostformer3.cost_heads_num = 1
# encoder
_CN.percostformer3.pretrain = True
_CN.percostformer3.del_layers = True
_CN.percostformer3.use_convertor = False
_CN.percostformer3.encoder_depth = 3
_CN.percostformer3.expand_factor = 4
_CN.percostformer3.vertical_encoder_attn = "twins"
_CN.percostformer3.attn_dim = 128
_CN.percostformer3.patch_size = 8
_CN.percostformer3.patch_embed = 'single'
_CN.percostformer3.cross_attn = "all"
_CN.percostformer3.gma = "GMA"
_CN.percostformer3.vert_c_dim = 64
_CN.percostformer3.cost_encoder_res = True
_CN.percostformer3.cnet = 'twins'
_CN.percostformer3.fnet = 'twins'
_CN.percostformer3.flow_or_pe = "and"
_CN.percostformer3.use_patch = False # use cost patch rather than local cost as query
_CN.percostformer3.use_rpe = False
_CN.percostformer3.detach_local = False
_CN.percostformer3.no_sc = False
_CN.percostformer3.r_16 = -1
_CN.percostformer3.quater_refine = False
# pretrain config
_CN.percostformer3.pretrain_mode = False
_CN.percostformer3.pic_size = [368, 496, 368, 496]
_CN.percostformer3.mask_ratio = 0.5
_CN.percostformer3.query_num = 30
_CN.percostformer3.no_border = True
_CN.percostformer3.gt_r = 15
_CN.percostformer3.fix_pe = False
# decoder
_CN.percostformer3.decoder_depth = 12
_CN.percostformer3.critical_params = ['vert_c_dim', 'encoder_depth', 'vertical_encoder_attn', "use_patch", "flow_or_pe", "use_rpe", "dropout", "flow_or_pe", "expand_factor"]

### TRAINER
_CN.trainer = CN()
_CN.trainer.scheduler = 'OneCycleLR'

_CN.trainer.optimizer = 'adamw'
_CN.trainer.canonical_lr = 25e-5
_CN.trainer.adamw_decay = 1e-4
_CN.trainer.clip = 1.0
_CN.trainer.num_steps = 120000
_CN.trainer.epsilon = 1e-8
_CN.trainer.anneal_strategy = 'linear'
def get_cfg():
    return _CN.clone()
