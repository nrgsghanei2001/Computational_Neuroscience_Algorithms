from pymonntorch import Behavior
import torch


class SynFun(Behavior):
	def initialize(self, sg):
		sg.W = sg.matrix(mode=0)
		sg.I = sg.dst.vector()
		self.is_inhibitory = self.parameter("is_inhibitory", False)

	def forward(self, sg):
		# print(sg.src.spike)
		sg.I = torch.sum(sg.W[sg.src.spike], axis=0)
		if self.is_inhibitory:
			sg.I *= -1


class InpSyn(Behavior):	
	def forward(self, ng):
		for syn in ng.afferent_synapses["All"]:
			ng.I += (syn.I / ng.N)
