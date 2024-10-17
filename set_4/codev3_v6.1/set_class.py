from collections import deque
import data_file



class sets():
	def __init__(self):
		#self.generated_veh = [deque([]) for _ in data_file.lanes]
		
		self.unspawned_veh = [deque([]) for _ in data_file.lanes]
		self.spawned_veh = [deque([]) for _ in data_file.lanes]
		
		#self.red_veh = [deque([]) for _ in data_file.lanes]
		#self.green_veh = [deque([]) for _ in data_file.lanes]
		#self.prior_red_veh = [deque([]) for _ in data_file.lanes]
		#self.prior_green_veh = [deque([]) for _ in data_file.lanes]
		#self.query_veh = [deque([]) for _ in data_file.lanes]
		self.done_veh = [deque([]) for _ in data_file.lanes]