from .base import BaseVN
from ..embed import BaseEmbed

class SpenctralVN(BaseVN):

    def __init__(self, multigraph=False, embed='ASE'):
        super().__init__(multigraph=multigraph)
        if issubclass(type(embed), BaseEmbed)