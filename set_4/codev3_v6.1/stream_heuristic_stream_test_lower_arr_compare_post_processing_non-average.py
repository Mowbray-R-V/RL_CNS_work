import os
import numpy as np
import csv
from matplotlib import pyplot as plt
import pickle

import data_file

#from multiprocessing import Pool




def stream_train_test():

	#num_train_iter = 36
	#num_sim = 10

	#train_iter_list = range(num_train_iter+1)

	train_iter_list = list(range(0, 1000, 100)) + list(range(1000, 10000, 1000)) + list(range(10000, 30000, 2500)) + [29900] #+ list(range(20000, 26000, 2500)) 

	num_train_iter = len(train_iter_list)

	sim_list = list(range(1, 5)) + list(range(6, 11))

	num_sim = len(sim_list)

	comb_opt_data = {}
	comb_opt_throughput = {}
	comb_opt_data_file_path = f"./data/compare_files/non-homogeneous_traffic_dt_0.1/arr_0.01/"

	test_data = {}
	throughput = {}
	test_data_file_path = f"./data/test_inhomo_stream/"

	percentage_comparison_dict = {}
	throughput_ratio_dict = {}

	for sim in sim_list:
		comb_opt_data[sim] = 0
		comb_opt_throughput[sim] = 0

		with open(f"{comb_opt_data_file_path}coord_phase_info_0.01_{sim}.csv", newline='') as csvfile:
			reader = csv.reader(csvfile)
			for ind, row in enumerate(reader):
				temp = float(row[1])*float(row[3])
				comb_opt_data[sim] += temp

				if ind*data_file.t_opti >= data_file.max_sim_time:
					break
			csvfile.close()

		with open(f"{comb_opt_data_file_path}throughput_0.01_{sim}.csv", newline='') as csvfile:
			reader = csv.reader(csvfile)
			for row in reader:
				if float(row[0]) == data_file.max_sim_time:
					comb_opt_throughput[sim] = float(row[1])
			csvfile.close()





	train_sim_list = list(range(1,11)) # list(range(1,4)) + [5] + list(range(8,11))

	for train_iter in train_iter_list:
		test_data[train_iter] = {}
		throughput[train_iter] = {}
		for train_sim in train_sim_list:
			test_data[train_iter][train_sim] = {}
			throughput[train_iter][train_sim] = {}
			for sim in sim_list:
				test_data[train_iter][train_sim][sim] = 0
				throughput[train_iter][train_sim][sim] = 0

				with open(f"{test_data_file_path}train_sim_{train_sim}/train_iter_{train_iter}/sim_{sim}/coord_phase_info.csv", newline='') as csvfile:
					reader = csv.reader(csvfile)
					#print(len(list(reader)))
					
					for row_ind, row in enumerate(reader):
						if row_ind == 0:
							temp_junk_num = float(row[1])
							temp_junk_val = float(row[3])
						temp = float(row[1])*float(row[3])
						#print(row_ind, temp)
						if (row_ind != 0) and (temp_junk_val == float(row[3])) and (temp_junk_num == float(row[1])):
							break

						#print(row_ind)
						test_data[train_iter][train_sim][sim] += temp

					csvfile.close()

				with open(f"{test_data_file_path}train_sim_{train_sim}/train_iter_{train_iter}/sim_{sim}/throughput_0.01_{sim}_train_iter_{train_iter}.csv", newline='') as csvfile:
					reader = csv.reader(csvfile)
					for row in reader:
						if float(row[0]) == data_file.max_sim_time:
							throughput[train_iter][train_sim][sim] = float(row[1])
					csvfile.close()



	for train_iter in train_iter_list:
		percentage_comparison_dict[train_iter] = {}
		throughput_ratio_dict[train_iter] = {}
		for train_sim in train_sim_list:
			percentage_comparison_dict[train_iter][train_sim] = {}
			throughput_ratio_dict[train_iter][train_sim] = {}
			for sim in sim_list:
				percentage_comparison_dict[train_iter][train_sim][sim] = 100*((comb_opt_data[sim] - test_data[train_iter][train_sim][sim])/(comb_opt_data[sim]))
				throughput_ratio_dict[train_iter][train_sim][sim] = comb_opt_throughput[sim]/throughput[train_iter][train_sim][sim]



	average_percentage_comparison_list = [0 for _ in range(num_train_iter)]

	var_percentage_comparison_list = [0 for _ in range(num_train_iter)]

	average_throughput_ratio_list = [0 for _ in range(num_train_iter)]

	var_throughput_ratio_list = [0 for _ in range(num_train_iter)] 


	for train_iter_ind, train_iter in enumerate(train_iter_list):
		temp_var_list = []
		temp_throughput_ratio = []

		for train_sim in train_sim_list:
			for sim in sim_list:
				temp_var_list.append(percentage_comparison_dict[train_iter][train_sim][sim])
				temp_throughput_ratio.append(throughput_ratio_dict[train_iter][train_sim][sim])
				average_percentage_comparison_list[train_iter_ind] += percentage_comparison_dict[train_iter][train_sim][sim]
				average_throughput_ratio_list[train_iter_ind] += throughput_ratio_dict[train_iter][train_sim][sim]

		var_percentage_comparison_list[train_iter_ind] = np.var(np.asarray(temp_var_list))
		var_throughput_ratio_list[train_iter_ind] = np.var(np.asarray(temp_throughput_ratio))

		average_percentage_comparison_list[train_iter_ind] = average_percentage_comparison_list[train_iter_ind]/(num_sim*len(train_sim_list))
		average_throughput_ratio_list[train_iter_ind] = average_throughput_ratio_list[train_iter_ind]/(num_sim*len(train_sim_list))


	plt.plot(train_iter_list, average_percentage_comparison_list)

	plt.title(f"averaged %opt-gap over {len(train_sim_list)} training intializations and {num_sim} test streams of 500s each", fontsize=18)

	plt.xlabel("number of training iterations (1 training iteration per 1 second simulation time)", fontsize=18)

	plt.ylabel(f"average %opt-gap", fontsize=18)

	#plt.plot(train_iter_list, var_percentage_comparison_list)


	plt.show()

	print(average_percentage_comparison_list[-1])

	plt.clf()

	plt.plot(train_iter_list, average_throughput_ratio_list)

	plt.title(f"averaged throughput ratio (wrt combined optimization) over {len(train_sim_list)} training intializations and {num_sim} test streams of 500s each", fontsize=18)

	plt.xlabel("number of training iterations (1 training iteration per 1 second simulation time)", fontsize=18)

	plt.ylabel("averaged of throughput ratio of RL agent v/s combined optimization", fontsize=18)

	plt.show()

	plt.clf()
	plt.plot(train_iter_list, var_throughput_ratio_list)

	plt.show()




def stream_norm_dist_and_vel_test_using_pickobj(_tr_iter_):

	#num_train_iter = 36
	#num_sim = 10

	#train_iter_list = range(num_train_iter+1)

	#train_iter_list = list(range(0, 1000, 100)) + list(range(1000, 10000, 1000)) + list(range(10000, 30000, 2500)) + [29900] #+ list(range(20000, 26000, 2500)) 

	# train_iter_list = list(range(0, 1000, 100)) + list(range(1000, 6000, 1000)) # + list(range(10000, 50000, 5000)) + [49900]

	#list(range(0, 1000, 100)) + list(range(1000, 10000, 1000)) + list(range(10000, 27400, 2500)) + [29900] #list(range(20000, 30000, 2500)) + list(range(30000, 60000, 5000)) + [59900]

	avg_time_to_cross_norm_dist_and_vel_all_arr = []

	avg_time_to_cross_comb_opt_all_arr = []

	avg_obj_fun_norm_dist_and_vel_all_arr = []

	avg_obj_fun_comb_opt_all_arr = []
 
	avg_frac_veh_crossed = []

	sp_t_limit = 300


	avg_true_arr_rate = []

	heuristic = data_file.used_heuristic

	write_path = f"../data/{heuristic}"

	if heuristic == None:
		write_path = f"../data/"

	arr_rate_times_100_array =     data_file.arr_rates_to_simulate #list(range(1, 11)) #+ list(range(20, 100, 10))

	for arr_rate in arr_rate_times_100_array:

		# arr_rate = arr_rate_times_100_array[arr_rate_ind]

		# if arr_rate_ind < 20:
		# 	arr_rate = round(arr_rate_ind*0.01, 2)

		# else:``````
		# 	arr_rate = round(arr_rate_ind*0.01, 1)			

		train_iter_list = [_tr_iter_]

		num_train_iter = len(train_iter_list)

		sim_list = list(range(1,2)) # + list(range(3,6)) + list(range(7, 10))
		train_sim_list =  list(range(1,2)) # + list(range(3,8)) + list(range(9, 10))# list(range(1,2))#list(range(1,3))+list(range(4,8)) +[9]#list(range(1,4)) + [5] + list(range(8,11))


################################################### edited ######################
		num_sim = len(sim_list)

		comb_opt_data = {}
		comb_opt_throughput = {}
		comb_opt_ttc = {}
		comb_opt_exit_vel_dict = {}
		comb_opt_data_file_path = f"../data/compare_files/homogeneous_traffic/arr_{arr_rate}/"

		test_data = {}
		throughput = {}
		ttc = {}
		fraction_of_robots_crossed = {}
		all_ttc = []
		exit_vel = {}

		percentage_comparison_dict = {}
		throughput_ratio_dict = {}

		total_comb_opt_veh = {}

		comb_opt_veh_dict = {}

		total_veh_num = {}
		heuristic_veh_dict = {}
		for train_iter in train_iter_list:
			test_data[train_iter] = {}
			throughput[train_iter] = {}
			ttc[train_iter] = {}
			exit_vel[train_iter] = {}
			fraction_of_robots_crossed[train_iter] = {}
			total_veh_num[train_iter] = {}
			for train_sim in train_sim_list:
				test_data[train_iter][train_sim] = {}
				throughput[train_iter][train_sim] = {}
				ttc[train_iter][train_sim] = {}
				exit_vel[train_iter][train_sim] = {}
				total_veh_num[train_iter][train_sim] = {}
				fraction_of_robots_crossed[train_iter][train_sim] = {}
				for sim in sim_list:

					if heuristic == None:
						test_data_file_path =  f"../data_version/version_2/arr_{arr_rate}/test_homo_stream/train_sim_{train_sim}/train_iter_{train_iter}"    #f"../data_version/version_1/arr_0.1/test_homo_stream/train_sim_1/train_iter_500"     #
					else: pass
					
					test_data[train_iter][train_sim][sim] = 0
					throughput[train_iter][train_sim][sim] = 0
					ttc[train_iter][train_sim][sim] = 0
					exit_vel[train_iter][train_sim][sim] = 0
					total_veh_num[train_iter][train_sim][sim] = 0
					fraction_of_robots_crossed[train_iter][train_sim][sim] = 0

					veh_num = 0
					veh_num_crossed = 0
					temp = 0
					temp_ttc = 0
					temp_exit_vel = 0
					veh_num_avoid = 0

					#print(f"out here: {test_data_file_path}train_sim_{train_sim}/train_iter_{train_iter}/pickobj_sim_{sim}")

					for c in os.listdir(f"{test_data_file_path}/pickobj_sim_{sim}"):
						#print("in here")
						try:
							file = open(f"{test_data_file_path}/pickobj_sim_{sim}/{c}",'rb')
							object_file = pickle.load(file)
							# try:
							# 	print(f"{object_file[int(c)].sp_t}", end='\r')
							# except:
							# 	print(f"\n{c}:{object_file.keys()}\n")
							# 	print(a)
							# print(f"veh_id: {object_file[int(c)].id}\tveh_p_traj: {object_file[int(c)].p_traj}")
							file.close()
						except:
							# print(file)
							continue

						if (object_file[int(c)].sp_t > sp_t_limit) or (object_file[int(c)].sp_t < 90.5):
							#print("***", object_file[int(c)].id)
							veh_num_avoid += 1
							continue
						
						else:
							
							#print("increment")
							try:
								# if object_file[int(c)].sp_t < 90: 
								# 	continue   
								index_var = 0
								# heuristic_veh_dict[c] = (object_file[int(c)].p_traj[int(data_file.T_sc/data_file.dt)] - object_file[int(c)].p0)
								# print(f'{object_file[int(c)].id}********')     #(f"veh: {c}\tdiff: {comb_opt_veh_dict[c] - heuristic_veh_dict[c]}")
								veh_num += 1
								for time, pos in zip(object_file[int(c)].t_ser, object_file[int(c)].p_traj):
									# print(f'inside---ID****: {object_file[int(c)].id },time:{time},pos:{pos}')			
									if pos >= object_file[int(c)].length :
										temp += object_file[int(c)].priority * (object_file[int(c)].p_traj[int(data_file.T_sc/data_file.dt) -1] - object_file[int(c)].p0)
										veh_num_crossed += 1
										temp_ttc += (time - object_file[int(c)].t_ser[0]) # object_file[int(c)].priority * 
										all_ttc.append(time - object_file[int(c)].t_ser[0])
										throughput[train_iter][train_sim][sim] += 1
										temp_exit_vel += object_file[int(c)].v_traj[index_var]
										break

									index_var += 1
							
							except IndexError:
								# print(f"index IndexError")
								continue
								#temp += object_file[int(c)].p_traj[-1]
								#temp_ttc += object_file[int(c)].t_ser[-1] - object_file[int(c)].t_ser[0]
								#pos = object_file[int(c)].p_traj[-1]

								#while pos < object_file[int(c)].length + object_file[int(c)].intsize:
								#	temp_ttc += data_file.dt
								#	pos += object_file[int(c)].v_traj[-1]*data_file.dt
					# print(f"veh num in sim {sim} is {veh_num}")
					print(f'********num_veh:{veh_num}, avoided:{veh_num_avoid}, crossed:{veh_num_crossed}, sim:{sim},arr:{arr_rate}, train_sim:{train_sim}, iter:{train_iter},ver:{10}')

					#exit()
     
     
     
					total_veh_num[train_iter][train_sim][sim] = veh_num
					#print(f'simulatio_step:{sim}')
					#print(f"heuristic: {heuristic}, arr_rate: {arr_rate}, train_sim: {train_sim}, train_iter:{train_iter} sim: {sim}, veh_num: {veh_num}, ...................") #, end="\r") # 
					test_data[train_iter][train_sim][sim] += temp
     
					fraction_of_robots_crossed[train_iter][train_sim][sim] = veh_num_crossed/total_veh_num[train_iter][train_sim][sim]

					print(type(total_veh_num[train_iter][train_sim][sim]))
					print(f"veh:{veh_num}, dem:{veh_num_crossed}, dem_1: {total_veh_num[train_iter][train_sim][sim]}, NR: {temp_ttc}, ...................") #, end="\r") # 
					ttc[train_iter][train_sim][sim] += temp_ttc/(veh_num_crossed)
					exit_vel[train_iter][train_sim][sim] += temp_exit_vel/total_veh_num[train_iter][train_sim][sim]

						#except:
						#	print(f"train_iter: {train_iter}, train_sim: {train_sim}, sim: {sim}...................", end="\r")
						#	continue
					




					#with open(f"{test_data_file_path}train_sim_{train_sim}/train_iter_{train_iter}/sim_{sim}/throughput_0.01_{sim}_train_iter_{train_iter}.csv", newline='') as csvfile:
					#	reader = csv.reader(csvfile)
					#	for row in reader:
					#		if float(row[0]) == data_file.max_sim_time:
					#			throughput[train_iter][train_sim][sim] = float(row[1])
					#	csvfile.close()

					# print(f"comb_opt_for sim_{sim}: {comb_opt_data[sim]}, heuristic: {test_data[train_iter][train_sim][sim]}")
		temp_obj_list = []
		temp_arr_list = []
		temp_fract_veh_crossed = []
		for train_iter in train_iter_list: 
			for train_sim in train_sim_list:
				for sim in sim_list:
					temp_obj_list.append(test_data[train_iter][train_sim][sim])
					temp_arr_list.append(total_veh_num[train_iter][train_sim][sim])
					temp_fract_veh_crossed.append(fraction_of_robots_crossed[train_iter][train_sim][sim])
     

		avg_obj_fun_norm_dist_and_vel_all_arr.append( np.average(np.asarray(temp_obj_list)) )
		avg_true_arr_rate.append(np.average(np.asarray(temp_arr_list)))
		avg_frac_veh_crossed.append(np.average(np.asarray(temp_fract_veh_crossed)))

		# avg_time_to_cross_norm_dist_and_vel_all_arr.append( sum([(ttc[train_iter][train_sim][sim_]/(len(sim_list)*total_veh_num[train_iter][train_sim][sim_])) for sim_ in sim_list]) )

		for train_iter in train_iter_list:
			percentage_comparison_dict[train_iter] = {}
			throughput_ratio_dict[train_iter] = {}
			for train_sim in train_sim_list:
				percentage_comparison_dict[train_iter][train_sim] = {}
				throughput_ratio_dict[train_iter][train_sim] = {}
				for sim in sim_list:
					# percentage_comparison_dict[train_iter][train_sim][sim] = 100*((comb_opt_data[sim] - test_data[train_iter][train_sim][sim])/(comb_opt_data[sim]))
					# throughput_ratio_dict[train_iter][train_sim][sim] = comb_opt_throughput[sim]/throughput[train_iter][train_sim][sim]
					...


		average_percentage_comparison_list = [0 for _ in range(num_train_iter)]

		var_percentage_comparison_list = [0 for _ in range(num_train_iter)]

		average_throughput_ratio_list = [0 for _ in range(num_train_iter)]

		var_throughput_ratio_list = [0 for _ in range(num_train_iter)]

		average_exit_vel_list = [0 for _ in range(num_train_iter)]

		var_exit_vel_list = [0 for _ in range(num_train_iter)]

		average_ttc_list = [0 for _ in range(num_train_iter)]

		var_ttc_list = [0 for _ in range(num_train_iter)]


		for train_iter_ind, train_iter in enumerate(train_iter_list):
			temp_var_list = []
			temp_throughput_ratio = []
			temp_ttc_list = []
			temp_exit_vel_list = []

			for train_sim in train_sim_list:
				for sim in sim_list:
					# temp_var_list.append(percentage_comparison_dict[train_iter][train_sim][sim])
					# temp_throughput_ratio.append(throughput_ratio_dict[train_iter][train_sim][sim])
					temp_ttc_list.append(ttc[train_iter][train_sim][sim])
					# temp_exit_vel_list.append(exit_vel[train_iter][train_sim][sim])
					# average_percentage_comparison_list[train_iter_ind] += percentage_comparison_dict[train_iter][train_sim][sim]
					# average_throughput_ratio_list[train_iter_ind] += throughput_ratio_dict[train_iter][train_sim][sim]
					average_ttc_list[train_iter_ind] += ttc[train_iter][train_sim][sim]
					# average_exit_vel_list[train_iter_ind] += exit_vel[train_iter][train_sim][sim]

			# var_percentage_comparison_list[train_iter_ind] = np.var(np.asarray(temp_var_list))
			# var_throughput_ratio_list[train_iter_ind] = np.var(np.asarray(temp_throughput_ratio))
			var_ttc_list[train_iter_ind] = np.var(np.asarray(temp_ttc_list))
			# var_exit_vel_list[train_iter_ind] = np.var(np.asarray(temp_exit_vel_list))


			# average_percentage_comparison_list[train_iter_ind] = average_percentage_comparison_list[train_iter_ind]/(num_sim*len(train_sim_list))
			# average_throughput_ratio_list[train_iter_ind] = average_throughput_ratio_list[train_iter_ind]/(num_sim*len(train_sim_list))
			average_ttc_list[train_iter_ind] = average_ttc_list[train_iter_ind]/(num_sim*len(train_sim_list))
			# average_exit_vel_list[train_iter_ind] = average_exit_vel_list[train_iter_ind]/(num_sim*len(train_sim_list))

		avg_time_to_cross_norm_dist_and_vel_all_arr.append(average_ttc_list)


		print(f"90%le ttc fir arr {arr_rate} is {np.percentile(np.asarray(all_ttc), 90)}")


	print(f"comb_opt_avg_obj_data: {avg_obj_fun_comb_opt_all_arr}")
	print(f"{heuristic}_avg_obj_data: {avg_obj_fun_norm_dist_and_vel_all_arr}")
	print(f"{heuristic}_avg_true_arr_rate: {avg_true_arr_rate}")
	print(f"average % diff: {average_percentage_comparison_list[train_iter_ind]}")
	""" 
	with open(f"../data/comb_avg_obj_fun_all_arr.csv", "w", newline="") as f:
		writer = csv.writer(f)
		#for j in train_iter_list:
		writer.writerows([[_] for _ in avg_obj_fun_comb_opt_all_arr])

	with open(f"{write_path}rl_avg_obj_fun_all_arr_{_tr_iter_}.csv", "w", newline="") as f:
		writer = csv.writer(f)
		#for j in train_iter_list:
		writer.writerows([[_] for _ in avg_obj_fun_norm_dist_and_vel_all_arr])

	with open(f"{write_path}rl_avg_frac-veh-crossed_all_arr_{_tr_iter_}.csv", "w", newline="") as f:
		writer = csv.writer(f)
		#for j in train_iter_list:
		writer.writerows([[_] for _ in avg_frac_veh_crossed])


	with open(f"{write_path}rl_avg_true_arr_rate_all_arr_{_tr_iter_}.csv", "w", newline="") as f:
		writer = csv.writer(f)
		#for j in train_iter_list:
		writer.writerows([[_] for _ in avg_true_arr_rate])


	"""



	print(f"comb_opt_avg_ttc_data: {avg_time_to_cross_comb_opt_all_arr}")
	print(f"{heuristic}_avg_ttc_data: {avg_time_to_cross_norm_dist_and_vel_all_arr}")
	print(f"{heuristic}_avg_frac-veh-crossed_data: {avg_frac_veh_crossed}")
	"""  

	with open(f"../data/comb_avg_ttc_all_arr.csv", "w", newline="") as f:
		writer = csv.writer(f)
		#for j in train_iter_list:
		writer.writerows([[_] for _ in avg_time_to_cross_comb_opt_all_arr])

	with open(f"{write_path}rl_avg_ttc_all_arr_{_tr_iter_}.csv", "w", newline="") as f:
		writer = csv.writer(f)
		#for j in train_iter_list:
		writer.writerows([[_[0]] for _ in avg_time_to_cross_norm_dist_and_vel_all_arr])

 	"""

if __name__ == "__main__":

	args_in = []
	iter_list = [500]#, 10000, 25000, 50000, 100000]#, 100000] #[2000, 5000, 12000] # [7000, 10000, 13000, 14000, 25000, 50000] #[0, 100, 500, 1000, 2000, 3000, 4000, 5000]
	for tr_iter in iter_list:
		args_in.append(tr_iter)
		stream_norm_dist_and_vel_test_using_pickobj(tr_iter)

