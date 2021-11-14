from skgframework.encoders.transformer_encoder import TransformerEncoder
from skgframework.encoders.rnn_encoder import RnnEncoder
from skgframework.encoders.rnn_encoder import LstmEncoder
from skgframework.encoders.rnn_encoder import GruEncoder
from skgframework.encoders.rnn_encoder import BirnnEncoder
from skgframework.encoders.rnn_encoder import BilstmEncoder
from skgframework.encoders.rnn_encoder import BigruEncoder
from skgframework.encoders.cnn_encoder import GatedcnnEncoder


str2encoder = {"transformer": TransformerEncoder, "rnn": RnnEncoder, "lstm": LstmEncoder,
               "gru": GruEncoder, "birnn": BirnnEncoder, "bilstm": BilstmEncoder, "bigru": BigruEncoder,
               "gatedcnn": GatedcnnEncoder}

__all__ = ["TransformerEncoder", "RnnEncoder", "LstmEncoder", "GruEncoder", "BirnnEncoder",
           "BilstmEncoder", "BigruEncoder", "GatedcnnEncoder", "str2encoder"]

