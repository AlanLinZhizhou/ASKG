from skgframework.targets.mlm_target import MlmTarget
from skgframework.targets.lm_target import LmTarget
from skgframework.targets.bert_target import BertTarget
from skgframework.targets.cls_target import ClsTarget
from skgframework.targets.bilm_target import BilmTarget
from skgframework.targets.albert_target import AlbertTarget
from skgframework.targets.seq2seq_target import Seq2seqTarget
from skgframework.targets.t5_target import T5Target
from skgframework.targets.gsg_target import GsgTarget
from skgframework.targets.bart_target import BartTarget
from skgframework.targets.prefixlm_target import PrefixlmTarget


str2target = {"bert": BertTarget, "mlm": MlmTarget, "lm": LmTarget,
              "bilm": BilmTarget, "albert": AlbertTarget, "seq2seq": Seq2seqTarget,
              "t5": T5Target, "gsg": GsgTarget, "bart": BartTarget,
              "cls": ClsTarget, "prefixlm": PrefixlmTarget}

__all__ = ["BertTarget", "MlmTarget", "LmTarget", "BilmTarget", "AlbertTarget",
           "Seq2seqTarget", "T5Target", "GsgTarget", "BartTarget", "ClsTarget", "PrefixlmTarget", "str2target"]
