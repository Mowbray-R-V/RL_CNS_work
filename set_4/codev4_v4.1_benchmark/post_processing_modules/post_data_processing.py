import os
import numpy as np
import csv
from matplotlib import pyplot as plt
import pickle
import data_file
import csv
import post_data_processing_utils as csv_print
import sys

#from multiprocessing import Pool




def stream_norm_dist_and_vel_test_using_pickobj(_tr_iter_, version):

	#num_train_iter = 36
	#num_sim = 10

	#train_iter_list = range(num_train_iter+1)

	#train_iter_list = list(range(0, 1000, 100)) + list(range(1000, 10000, 1000)) + list(range(10000, 30000, 2500)) + [29900] #+ list(range(20000, 26000, 2500)) 

	# train_iter_list = list(range(0, 1000, 100)) + list(range(1000, 6000, 1000)) # + list(range(10000, 50000, 5000)) + [49900]

	#list(range(0, 1000, 100)) + list(range(1000, 10000, 1000)) + list(range(10000, 27400, 2500)) + [29900] #list(range(20000, 30000, 2500)) + list(range(30000, 60000, 5000)) + [59900]

	avg_time_to_cross_norm_dist_and_vel_all_arr = []
	avg_tb_time= []
	avg_ta_time= []
	avg_time_to_cross_norm_dist_and_vel_all_arr_prior = []

	avg_time_to_cross_comb_opt_all_arr = []

	avg_obj_fun_norm_dist_and_vel_all_arr = []

	avg_obj_fun_comb_opt_all_arr = []
 
	avg_frac_veh_crossed = []
	avg_nb = []


	sp_t_limit = 450
	sp_t_start = 25


	avg_true_arr_rate = []
	avg_true_thro_rate = []

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

		sim_list = list(range(1,51)) # + list(range(3,6)) + list(range(7, 10))
		train_sim_list =  list(range(1,2)) # + list(range(3,8)) + list(range(9, 10))# list(range(1,2))#list(range(1,3))+list(range(4,8)) +[9]#list(range(1,4)) + [5] + list(range(8,11))


################################################### edited ######################
		num_sim = len(sim_list)

		comb_opt_data = {}
		comb_opt_throughput = {}
		comb_opt_ttc = {}
		comb_opt_exit_vel_dict = {}
		comb_opt_data_file_path = f"../data/compare_files/homogeneous_traffic/arr_{arr_rate}/"

		test_data = {}
		throughput_a = {}
		throughput_b = {}
		ttc = {}
		ta = {}
		tb = {}
		ttc_prior = {}
		fraction_of_robots_crossed = {}
		all_ttc_a = []
		all_ttc_b = []
		exit_vel = {}

		percentage_comparison_dict = {}
		throughput_ratio_dict = {}

		total_comb_opt_veh = {}

		comb_opt_veh_dict = {}

		total_veh_num = {}
		total_veh_num_cross = {}
		heuristic_veh_dict = {}
		for train_iter in train_iter_list:
			test_data[train_iter] = {}
			throughput_a[train_iter] = {}
			throughput_b[train_iter] = {}
			ttc[train_iter] = {}
			ta[train_iter] = {}
			tb[train_iter] = {}
			ttc_prior[train_iter] = {}
			exit_vel[train_iter] = {}
			fraction_of_robots_crossed[train_iter] = {}
			total_veh_num[train_iter] = {}
			total_veh_num_cross[train_iter] = {}
			for train_sim in train_sim_list:
				test_data[train_iter][train_sim] = {}
				throughput_a[train_iter][train_sim] = {}
				throughput_b[train_iter][train_sim] = {}
				ttc[train_iter][train_sim] = {}
				ta[train_iter][train_sim] = {}
				ta[train_iter][train_sim] = {}
				tb[train_iter][train_sim] = {}
				ttc_prior[train_iter][train_sim] = {}
				exit_vel[train_iter][train_sim] = {}
				total_veh_num[train_iter][train_sim] = {}
				total_veh_num_cross[train_iter][train_sim] = {}
				fraction_of_robots_crossed[train_iter][train_sim] = {}
				for sim in sim_list:

					if heuristic == None:
						test_data_file_path =  f"../data_version/version_{int(version)}/arr_{arr_rate}/test_homo_stream/train_sim_{train_sim}/train_iter_{train_iter}"    #f"../data_version/version_1/arr_0.1/test_homo_stream/train_sim_1/train_iter_500"     #
					else: pass
					
					test_data[train_iter][train_sim][sim] = 0
					throughput_a[train_iter][train_sim][sim] = 0
					throughput_b[train_iter][train_sim][sim] = 0
					ttc[train_iter][train_sim][sim] = 0
					ta[train_iter][train_sim][sim] = 0
					tb[train_iter][train_sim][sim] = 0
					ttc_prior[train_iter][train_sim][sim] = 0
					exit_vel[train_iter][train_sim][sim] = 0
					total_veh_num[train_iter][train_sim][sim] = 0
					total_veh_num_cross[train_iter][train_sim][sim] = 0
					fraction_of_robots_crossed[train_iter][train_sim][sim] = 0

					veh_num = 0
					veh_num_crossed = 0
					temp = 0
					temp_ttc_a = 0
					temp_ttc_b = 0
					temp_ttc_prior = 0
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

						if (object_file[int(c)].sp_t > sp_t_limit) or (object_file[int(c)].sp_t <sp_t_start ):
							#print("***", object_file[int(c)].id)
							veh_num_avoid += 1
							continue
						
						else:
							
							#print("increment")
							try:
								# if object_file[int(c)].sp_t < 90: 
								# 	continue   
								index_var = 0
								inter = 0
								# heuristic_veh_dict[c] = (object_file[int(c)].p_traj[int(data_file.T_sc/data_file.dt)] - object_file[int(c)].p0)
								# print(f'{object_file[int(c)].id}********')     #(f"veh: {c}\tdiff: {comb_opt_veh_dict[c] - heuristic_veh_dict[c]}")
								veh_num += 1
								for time, pos in zip(object_file[int(c)].t_ser, object_file[int(c)].p_traj):
									# print(f'inside---ID****: {object_file[int(c)].id },time:{time},pos:{pos}')			
									if pos >= object_file[int(c)].length : #and time<=sp_t_limit:
										temp += object_file[int(c)].priority * (object_file[int(c)].p_traj[int(data_file.T_sc/data_file.dt) -1] - object_file[int(c)].p0)
										veh_num_crossed += 1
										temp_ttc_a += (time - object_file[int(c)].t_ser[0]) # object_file[int(c)].priority * 
										temp_ttc_prior += (time - object_file[int(c)].t_ser[0])*object_file[int(c)].priority 
										all_ttc_a.append(time - object_file[int(c)].t_ser[0])
										throughput_a[train_iter][train_sim][sim] += 1
										temp_exit_vel += object_file[int(c)].v_traj[index_var]
										inter  = 1
										break
									index_var += 1
								# print(f'inter:{inter}----------------')	

								if inter==0:
									# print(f'inside---ID****: {object_file[int(c)].id },time:{time},pos:{pos}---------------------------------`')			
									if pos < object_file[int(c)].length :
										temp_ttc_b += (500 - object_file[int(c)].sp_t)  
										# temp_ttc_prior += (sp_t_limit - time)*object_file[int(c)].priority 
										all_ttc_b.append(500 - sp_t_limit)
										throughput_b[train_iter][train_sim][sim] += 1


							except IndexError:
								# print(f"index IndexError")
								continue
								#temp += object_file[int(c)].p_traj[-1]
								#temp_ttc_a += object_file[int(c)].t_ser[-1] - object_file[int(c)].t_ser[0]
								#pos = object_file[int(c)].p_traj[-1]

								#while pos < object_file[int(c)].length + object_file[int(c)].intsize:
								#	temp_ttc_a += data_file.dt
								#	pos += object_file[int(c)].v_traj[-1]*data_file.dt
					# print(f"veh num in sim {sim} is {veh_num}")
					# print(f'********num_veh:{veh_num}, avoided:{veh_num_avoid}, crossed:{veh_num_crossed}, sim:{sim},arr:{arr_rate}, train_sim:{train_sim}, iter:{train_iter},ver:{10}')

					#exit()
     
     
     
					total_veh_num[train_iter][train_sim][sim] = veh_num
					total_veh_num_cross[train_iter][train_sim][sim] = veh_num_crossed
					#print(f'simulatio_step:{sim}')
					#print(f"heuristic: {heuristic}, arr_rate: {arr_rate}, train_sim: {train_sim}, train_iter:{train_iter} sim: {sim}, veh_num: {veh_num}, ...................") #, end="\r") # 
					test_data[train_iter][train_sim][sim] += temp
     
					fraction_of_robots_crossed[train_iter][train_sim][sim] = total_veh_num[train_iter][train_sim][sim]/veh_num_crossed

					print(type(total_veh_num[train_iter][train_sim][sim]))
					print(f"veh:{veh_num}, dem:{veh_num_crossed}, dem_1: {total_veh_num[train_iter][train_sim][sim]}, NR: {temp_ttc_a}, ...................") #, end="\r") # 
					ta[train_iter][train_sim][sim] += temp_ttc_a
					ttc[train_iter][train_sim][sim] += temp_ttc_a/(veh_num_crossed)
					print(f' TTC values:train_iter:{train_iter}, train_Sim:{train_sim}, sim:{sim}, TTC:{ttc[train_iter][train_sim][sim]}')
					tb[train_iter][train_sim][sim] += temp_ttc_b
					ttc_prior[train_iter][train_sim][sim] += temp_ttc_prior/(veh_num_crossed)
					exit_vel[train_iter][train_sim][sim] += temp_exit_vel/total_veh_num[train_iter][train_sim][sim]

						#except:
						#	print(f"train_iter: {train_iter}, train_sim: {train_sim}, sim: {sim}...................", end="\r")
						#	continue
					
					print(f'*****sim::{sim},ta:{ta[train_iter][train_sim][sim] }, ttc:{ttc[train_iter][train_sim][sim]}, tb:{tb[train_iter][train_sim][sim]}, veh_num:{veh_num}, veh_crossed:{veh_num_crossed}')




					#with open(f"{test_data_file_path}train_sim_{train_sim}/train_iter_{train_iter}/sim_{sim}/throughput_0.01_{sim}_train_iter_{train_iter}.csv", newline='') as csvfile:
					#	reader = csv.reader(csvfile)
					#	for row in reader:
					#		if float(row[0]) == data_file.max_sim_time:
					#			throughput_a[train_iter][train_sim][sim] = float(row[1])
					#	csvfile.close()

					# print(f"comb_opt_for sim_{sim}: {comb_opt_data[sim]}, heuristic: {test_data[train_iter][train_sim][sim]}")
		temp_obj_list = []
		temp_arr_list = []
		temp_thro_list = []
		temp_fract_veh_crossed = []
		nb =[]
		for train_iter in train_iter_list: 
			for train_sim in train_sim_list:
				for sim in sim_list:
					temp_obj_list.append(test_data[train_iter][train_sim][sim])
					temp_arr_list.append(total_veh_num[train_iter][train_sim][sim])
					temp_thro_list.append(total_veh_num_cross[train_iter][train_sim][sim])
					temp_fract_veh_crossed.append(fraction_of_robots_crossed[train_iter][train_sim][sim])
					nb.append(total_veh_num[train_iter][train_sim][sim]-total_veh_num_cross[train_iter][train_sim][sim])
     

		avg_obj_fun_norm_dist_and_vel_all_arr.append( np.average(np.asarray(temp_obj_list)) )
		avg_true_arr_rate.append(np.average(np.asarray(temp_arr_list)))
		avg_true_thro_rate.append(np.average(np.asarray(temp_thro_list)))
		avg_frac_veh_crossed.append(np.average(np.asarray(temp_fract_veh_crossed)))
		avg_nb.append(np.average(np.asarray(nb)))

		# avg_time_to_cross_norm_dist_and_vel_all_arr.append( sum([(ttc[train_iter][train_sim][sim_]/(len(sim_list)*total_veh_num[train_iter][train_sim][sim_])) for sim_ in sim_list]) )

		for train_iter in train_iter_list:
			percentage_comparison_dict[train_iter] = {}
			throughput_ratio_dict[train_iter] = {}
			for train_sim in train_sim_list:
				percentage_comparison_dict[train_iter][train_sim] = {}
				throughput_ratio_dict[train_iter][train_sim] = {}
				for sim in sim_list:
					# percentage_comparison_dict[train_iter][train_sim][sim] = 100*((comb_opt_data[sim] - test_data[train_iter][train_sim][sim])/(comb_opt_data[sim]))
					# throughput_ratio_dict[train_iter][train_sim][sim] = comb_opt_throughput[sim]/throughput_a[train_iter][train_sim][sim]
					...


		average_percentage_comparison_list = [0 for _ in range(num_train_iter)]

		var_percentage_comparison_list = [0 for _ in range(num_train_iter)]

		average_throughput_ratio_list = [0 for _ in range(num_train_iter)]

		var_throughput_ratio_list = [0 for _ in range(num_train_iter)]

		average_exit_vel_list = [0 for _ in range(num_train_iter)]

		var_exit_vel_list = [0 for _ in range(num_train_iter)]

		average_ttc_list = [0 for _ in range(num_train_iter)]
		average_tb_list = [0 for _ in range(num_train_iter)]
		average_ta_list = [0 for _ in range(num_train_iter)]

		average_ttc_list_prior = [0 for _ in range(num_train_iter)]

		var_ttc_list = [0 for _ in range(num_train_iter)]


		for train_iter_ind, train_iter in enumerate(train_iter_list):
			temp_var_list = []
			temp_throughput_ratio = []
			temp_ttc_list = []
			temp_tb_list = []
			temp_ta_list = []
			temp_ttc_prior_list = []
			temp_exit_vel_list = []

			for train_sim in train_sim_list:
				for sim in sim_list:
					# temp_var_list.append(percentage_comparison_dict[train_iter][train_sim][sim])
					# temp_throughput_ratio.append(throughput_ratio_dict[train_iter][train_sim][sim])
					temp_ttc_list.append(ttc[train_iter][train_sim][sim])
					temp_ta_list.append(ta[train_iter][train_sim][sim])
					temp_tb_list.append(tb[train_iter][train_sim][sim])
					temp_ttc_prior_list.append(ttc_prior[train_iter][train_sim][sim])
					# temp_exit_vel_list.append(exit_vel[train_iter][train_sim][sim])
					# average_percentage_comparison_list[train_iter_ind] += percentage_comparison_dict[train_iter][train_sim][sim]
					# average_throughput_ratio_list[train_iter_ind] += throughput_ratio_dict[train_iter][train_sim][sim]
					average_ttc_list[train_iter_ind] += ttc[train_iter][train_sim][sim]
					average_ta_list[train_iter_ind] += ta[train_iter][train_sim][sim]
					average_tb_list[train_iter_ind] += tb[train_iter][train_sim][sim]
					average_ttc_list_prior[train_iter_ind] += ttc_prior[train_iter][train_sim][sim]
					# average_exit_vel_list[train_iter_ind] += exit_vel[train_iter][train_sim][sim]

			# var_percentage_comparison_list[train_iter_ind] = np.var(np.asarray(temp_var_list))
			# var_throughput_ratio_list[train_iter_ind] = np.var(np.asarray(temp_throughput_ratio))
			var_ttc_list[train_iter_ind] = np.var(np.asarray(temp_ttc_prior_list))
			# var_exit_vel_list[train_iter_ind] = np.var(np.asarray(temp_exit_vel_list))


			# average_percentage_comparison_list[train_iter_ind] = average_percentage_comparison_list[train_iter_ind]/(num_sim*len(train_sim_list))
			# average_throughput_ratio_list[train_iter_ind] = average_throughput_ratio_list[train_iter_ind]/(num_sim*len(train_sim_list))
			average_ttc_list[train_iter_ind] = average_ttc_list[train_iter_ind]/(num_sim*len(train_sim_list))
			average_tb_list[train_iter_ind] = average_tb_list[train_iter_ind]/(num_sim*len(train_sim_list))
			average_ta_list[train_iter_ind] = average_ta_list[train_iter_ind]/(num_sim*len(train_sim_list))
			average_ttc_list_prior[train_iter_ind] = average_ttc_list_prior[train_iter_ind]/(num_sim*len(train_sim_list))

			# average_exit_vel_list[train_iter_ind] = average_exit_vel_list[train_iter_ind]/(num_sim*len(train_sim_list))

		avg_time_to_cross_norm_dist_and_vel_all_arr.append(average_ttc_list)
		avg_tb_time.append(average_tb_list)
		avg_ta_time.append(average_ta_list)
		avg_time_to_cross_norm_dist_and_vel_all_arr_prior.append(average_ttc_list_prior)


		print(f"90%le ttc fir arr {arr_rate} is {np.percentile(np.asarray(all_ttc_a), 90)}")


	# print(f"comb_opt_avg_obj_data: {avg_obj_fun_comb_opt_all_arr}")
	# print(f"{heuristic}_avg_obj_data: {avg_obj_fun_norm_dist_and_vel_all_arr}")
	print(f"avg_true_arr_rate: {avg_true_arr_rate[0]/(8*(sp_t_limit - sp_t_start))}")
	print(f"avg_true_throughput: {avg_true_thro_rate}")
	print(f"avg_true_throughput rate: {avg_true_thro_rate[0]/(8*(sp_t_limit - sp_t_start))}")
	# print(f"average % diff: {average_percentage_comparison_list[train_iter_ind]}")
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



	# print(f"comb_opt_avg_ttc_data: {avg_time_to_cross_comb_opt_all_arr}")
	print(f"avg_ttc_data: {avg_time_to_cross_norm_dist_and_vel_all_arr}")
	print(f"avg_ttc_data*prior: {avg_time_to_cross_norm_dist_and_vel_all_arr_prior}")
	print(f"crossing _index: {avg_frac_veh_crossed}")
	print(f"avg TA: {avg_ta_time}")
	print(f"avg NA: {avg_true_thro_rate[0]}")
	print(f"avg TB: {avg_tb_time}")
	print(f"avg NB: {avg_nb}")
	print(f"avg Tc: {avg_tb_time[0][0]  + avg_ta_time[0][0]}")
	print(f"avg Nc: {avg_nb[0]  + avg_true_thro_rate[0] }")



	# TTC
# TTC*priority
# Ta, Na
# Tb, Nb
# Tc, Nc
# Ratio of the above
# avg crossing index
# Avg arrival rate 
# avg throughput in evaluation time
# Avg throughput rate



	############write to a file ############
	csv_print.data_csv(
		avg_time_to_cross_norm_dist_and_vel_all_arr[0][0],        # ttc
		avg_time_to_cross_norm_dist_and_vel_all_arr_prior[0][0],  # ttcp
		avg_ta_time[0][0],                                        # ta
		avg_true_thro_rate[0],                               # na
		avg_tb_time[0][0],                                        # tb
		avg_nb[0],                                             # nb
		(avg_tb_time[0][0]  + avg_ta_time[0][0]),           # tc
		(avg_nb[0]  + avg_true_thro_rate[0]),               # nc
		((avg_tb_time[0][0]  + avg_ta_time[0][0])/(avg_nb[0]  + avg_true_thro_rate[0])),  # ratioc
		avg_frac_veh_crossed[0],								#crossing index
		(avg_true_arr_rate[0]/(8*(sp_t_limit - sp_t_start))),     # arr_rate
		avg_true_thro_rate[0],                                 # thro
		(avg_true_thro_rate[0]/(8*(sp_t_limit - sp_t_start)))     # thro_rate
	)
	############write to a file ############



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
	version = int(sys.argv[1]) 
	iter_list = [500]#, 10000, 25000, 50000, 100000]#, 100000] #[2000, 5000, 12000] # [7000, 10000, 13000, 14000, 25000, 50000] #[0, 100, 500, 1000, 2000, 3000, 4000, 5000]
	for tr_iter in iter_list:
		args_in.append(tr_iter)
		stream_norm_dist_and_vel_test_using_pickobj(tr_iter, version)

