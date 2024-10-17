from collections import deque
from utility import config



class sets():
	def __init__(self):
		#self.generated_veh = [deque([]) for _ in config.lanes]
		
		self.unspawned_veh = [deque([]) for _ in config.lanes]
		self.spawned_veh = [deque([]) for _ in config.lanes]
		
		#self.red_veh = [deque([]) for _ in config.lanes]
		#self.green_veh = [deque([]) for _ in config.lanes]
		#self.prior_red_veh = [deque([]) for _ in config.lanes]
		#self.prior_green_veh = [deque([]) for _ in config.lanes]
		#self.query_veh = [deque([]) for _ in config.lanes]
		self.done_veh = [deque([]) for _ in config.lanes]