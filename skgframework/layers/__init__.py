from skgframework.layers.embeddings import WordEmbedding
from skgframework.layers.embeddings import WordPosEmbedding
from skgframework.layers.embeddings import WordPosSegEmbedding
from skgframework.layers.embeddings import WordSinusoidalposEmbedding


str2embedding = {"word": WordEmbedding, "word_pos": WordPosEmbedding, "word_pos_seg": WordPosSegEmbedding,
                 "word_sinusoidalpos": WordSinusoidalposEmbedding}

__all__ = ["WordEmbedding", "WordPosEmbedding", "WordPosSegEmbedding",
           "WordSinusoidalposEmbedding", "str2embedding"]
