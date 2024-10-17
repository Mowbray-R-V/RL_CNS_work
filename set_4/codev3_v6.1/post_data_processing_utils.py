import os
import csv

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




def data_csv(ttc, ttcp, ta, na, tb, nb, tc ,nc, ratio_c, crossing_index, arr_rate, thro, thro_rate ): 


    if os.path.exists("sim_results_{__file__}.csv"):
        os.remove("sim_results_{__file__}.csv")

    data1 = [
        {'TTC': ttc, 'TTC*priority': ttcp, 'ta': ta, 'na': na, 'tb': tb, 'nb': nb, 'tc': tc, 'nc': nc, 'ratio_c': ratio_c, \
            'avg crossing index': crossing_index, 'Avg arrival rate ': arr_rate, 'avg throughput': thro, 'Avg throughput rate': thro_rate}
    ]

    def prepare_rows(data):
        return {
            'TTC': [entry['TTC'] for entry in data],
            'TTC*priority': [entry['TTC*priority'] for entry in data],
            'ta': [entry['ta'] for entry in data],
            'na': [entry['na'] for entry in data],
            'tb': [entry['tb'] for entry in data],
            'nb': [entry['nb'] for entry in data],
            'tc': [entry['tc'] for entry in data],
            'nc': [entry['nc'] for entry in data],

            'ratio_c': [entry['ratio_c'] for entry in data],
            'avg crossing index': [entry['avg crossing index'] for entry in data],
            'Avg arrival rate ': [entry['Avg arrival rate '] for entry in data],
            'avg throughput': [entry['avg throughput'] for entry in data],
            'Avg throughput rate': [entry['Avg throughput rate'] for entry in data]
        }

    rows = prepare_rows(data1)

    current_directory = os.getcwd()
    current_folder_name = os.path.basename(current_directory)

    with open(f'sim_results_{current_folder_name}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        for key, values in rows.items():
            writer.writerow([key] + values)
