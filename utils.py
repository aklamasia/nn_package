import sys

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '#' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %s%s ...%s/%s' % (bar, percents, '%', str(count), str(total)))
    sys.stdout.flush()
    
def average(lst):
    return sum(lst) / len(lst) 

def best_accouracy(lst):
    return max(lst)

def best_time(lst):
    return min(lst)

def calculate_result(algorithm_list, index):
    algorithm_list[index]['average_result_accouracy'] = format(average(algorithm_list[index]['data']['accouracy']), '.2f')
    algorithm_list[index]['average_result_time'] = format(average(algorithm_list[index]['data']['average_time']), '.5f')
    algorithm_list[index]['best_result_accouracy'] = format(best_accouracy(algorithm_list[index]['data']['accouracy']), '.2f')
    algorithm_list[index]['best_result_time'] = format(best_time(algorithm_list[index]['data']['average_time']), '.5f')
    
def writer_report(text_file, algorithm_list, index, kfold):
    text_file.write("\n"+algorithm_list[index]['label']+"\n")
    for x in range(kfold):
        text_file.write("Batch " + str(x+1))
        string = "\nresult > "
        for key, value in algorithm_list[index]['data'].items():
            label = key.replace('_',' ')
            suffix = "s " if "time" in label else  "% "
            if isinstance(value[x], float):
                val = format(value[x], '.5f') if "time" in label else format(value[x], '.2f')
            else:
                val = value[x]
            string = string + label + " = " + str(val) + suffix
        text_file.write(string+"\n")
    text_file.write("==========================================\n")
    text_file.write("avg result  :%s, %s\n" %(algorithm_list[index]['average_result_accouracy'], algorithm_list[index]['average_result_time']))
    text_file.write("best result :%s, %s\n" %(algorithm_list[index]['best_result_accouracy'], algorithm_list[index]['best_result_time']))
    text_file.write("==========================================\n")